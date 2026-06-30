import argparse
import hashlib
import json
import os
import pprint

import clip
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from datasets import utils
from datasets.dataset import EEG_Dataset
from modules.ViViTEmotionNet import create_model
from utils import Text_Prompt
from utils.trainer_entropy import Trainer as DefaultTrainer
from modules.cfbm.checkpoint_loader import load_checkpoint_with_audit


class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)


class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)


def _label_from_path(path):
    return int(os.path.splitext(path)[0].split("_")[-1])


def _count_labels(paths, num_classes):
    counts = [0 for _ in range(num_classes)]
    for path in paths:
        counts[_label_from_path(path)] += 1
    return counts


def _fingerprint(paths):
    digest = hashlib.sha256()
    for path in sorted(str(p).replace("\\", "/") for p in paths):
        digest.update(path.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def _audit_split(source_train, support, query, num_classes):
    source_set, support_set, query_set = set(source_train), set(support), set(query)
    return {
        "source_train_count": len(source_train),
        "support_count": len(support),
        "query_test_count": len(query),
        "train_total_count": len(source_train) + len(support),
        "source_label_counts": _count_labels(source_train, num_classes),
        "support_label_counts": _count_labels(support, num_classes),
        "query_label_counts": _count_labels(query, num_classes),
        "source_support_overlap": len(source_set & support_set),
        "source_query_overlap": len(source_set & query_set),
        "support_query_overlap": len(support_set & query_set),
        "all_overlap_zero": not (source_set & support_set or source_set & query_set or support_set & query_set),
        "fingerprint": _fingerprint(source_train + support + query),
    }


def _validate_eeg_topoencoder_config(config):
    data = config.get("data", {})
    network = config.get("network", {})
    if data.get("dataset") != "SEED":
        raise ValueError("clean SEED cross-subject entry requires data.dataset=SEED")
    if int(data.get("num_classes")) != 3:
        raise ValueError("clean SEED cross-subject entry requires data.num_classes=3")
    if network.get("feature") != "DE":
        raise ValueError("clean SEED cross-subject entry requires network.feature=DE")
    if int(network.get("image_channels")) != 6:
        raise ValueError("clean SEED cross-subject entry requires network.image_channels=6")
    if network.get("spectral_type") != "Transformer":
        raise ValueError("clean SEED cross-subject entry requires network.spectral_type=Transformer")
    if network.get("temporal_type") != "Transformer":
        raise ValueError("clean SEED cross-subject entry requires network.temporal_type=Transformer")
    if network.get("eeg_topoencoder_cfbm", {}).get("enabled", False):
        raise ValueError("clean SEED cross-subject entry forbids CFBM")
    mode = config.get("checkpoint_loader_mode", "from_scratch")
    pretrain = config.get("pretrain")
    if mode == "from_scratch":
        if config.get("isPretrain", False) or pretrain:
            raise ValueError("clean SEED cross-subject from_scratch entry must not set pretrain")
    elif mode == "seediv_eeg_topoencoder_pretrain":
        if config.get("isPretrain", False):
            raise ValueError("SEED cross-subject downstream config should not set isPretrain=true")
        if not pretrain:
            raise ValueError("seediv_eeg_topoencoder_pretrain mode requires a EEG-TopoEncoder checkpoint path")
    else:
        raise ValueError(f"clean SEED cross-subject entry refuses checkpoint_loader_mode={mode}")


def _build_split(config, test_subject, session):
    source_train, test_candidates = utils.get_cross_subject_train_test_filets_path(
        data_path=config["data"]["datasets_path"],
        num_trial=config["data"]["num_trial"],
        test_subjectId=test_subject,
        sessionId=session,
    )
    manifest_path = f"manifests/seed_stable_official/cross_subject_sess{session}_subject{test_subject}.json"
    support, query = utils.cross_subject_n_shot(
        test_candidates,
        num_shot=config["data"]["num_shot"],
        num_classes=config["data"]["num_classes"],
        seed=config["random_seed"],
        manifest_path=manifest_path,
        manifest_metadata={
            "fold_type": "official_cross_subject",
            "session": session,
            "test_subject": test_subject,
        },
        dataset=config["data"]["dataset"],
        session=session,
        test_subject=test_subject,
    )
    return source_train, support, query, manifest_path


def _make_models(config):
    device_ids = list(range(len(str(config["gpu_device_id"]).split(","))))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_clip, _ = clip.load("./clip_weights/ViT-B-16.pt", device=device)
    base_image_model = create_model(config)
    checkpoint_audit = load_checkpoint_with_audit(
        base_image_model,
        checkpoint_path=config.get("pretrain"),
        mode=config.get("checkpoint_loader_mode", "from_scratch"),
        strict_eeg_topoencoder=True,
    )
    print("EEG_TOPOENCODER_CROSS_SUBJECT_CHECKPOINT_AUDIT_JSON=" + json.dumps(checkpoint_audit, sort_keys=True))
    model_image = ImageCLIP(base_image_model)
    model_text = TextCLIP(model_clip)
    model_image = nn.DataParallel(model_image, device_ids=device_ids).cuda()
    model_text = nn.DataParallel(model_text, device_ids=device_ids).cuda()
    for param in model_text.parameters():
        param.requires_grad = False
    return model_image, model_text, model_clip, checkpoint_audit


def _select_trainer(config):
    etea_cgra_cfg = config.get("etea_cgra", {}) or {}
    if etea_cgra_cfg.get("enabled", False):
        from utils.trainer_etea_cgra import ETEACGRATrainer

        print("Using ETEACGRATrainer because etea_cgra.enabled=true")
        return ETEACGRATrainer
    return DefaultTrainer


def _run_dry_run(config, train_list, query_list, split_audit, session, test_subject):
    model_image, model_text, model_clip, checkpoint_audit = _make_models(config)
    temporal_calls = []

    def hook(_module, inputs, output):
        in_shape = list(inputs[0].shape) if inputs else None
        out_shape = list(output.shape) if hasattr(output, "shape") else None
        temporal_calls.append({"input": in_shape, "output": out_shape})

    hooks = []
    for name, module in model_image.named_modules():
        if "temporal" in name.lower():
            hooks.append(module.register_forward_hook(hook))

    dataset = EEG_Dataset(
        images_path=train_list,
        image_height=config["network"]["image_height"],
        image_width=config["network"]["image_width"],
        num_classes=config["data"]["num_classes"],
        feature=config["network"]["feature"],
        map_type="SST",
    )
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=2, shuffle=False, pin_memory=True, num_workers=0)
    images, labels = next(iter(loader))
    images = images.cuda().float()
    labels = labels.cuda()
    labels_idx = torch.argmax(labels, dim=1)

    classes, num_text_aug, _ = Text_Prompt.eeg_text_prompt(config["data"]["classes_names"])
    model_text.eval()
    model_image.eval()
    with torch.no_grad():
        text_features = F.normalize(model_text(classes.cuda()), dim=-1)
        image_features = F.normalize(model_image(images), dim=-1)
        logits = config["solver"].get("logit_scale", 1.0) * (image_features @ text_features.T)
        logits = logits.view(image_features.shape[0], num_text_aug, config["data"]["num_classes"]).mean(dim=1)
        loss = torch.nn.CrossEntropyLoss(reduction="sum")(logits, labels_idx)

    for item in hooks:
        item.remove()
    report = {
        "dataset": config["data"]["dataset"],
        "session": session,
        "test_subject": test_subject,
        "num_classes": config["data"]["num_classes"],
        "input_shape": list(images.shape),
        "logits_shape": list(logits.shape),
        "feature": config["network"]["feature"],
        "psd_present": False,
        "spectral_type": config["network"]["spectral_type"],
        "legoformer_used": False,
        "temporal_called": len(temporal_calls) > 0,
        "temporal_shapes": temporal_calls[:3],
        "checkpoint_mode": config.get("checkpoint_loader_mode", "from_scratch"),
        "checkpoint_path": config.get("pretrain"),
        "source_key_count": checkpoint_audit.get("source_key_count"),
        "target_key_count": checkpoint_audit.get("target_key_count"),
        "matched_key_count": checkpoint_audit.get("matched_key_count"),
        "loaded_parameter_ratio": checkpoint_audit.get("loaded_parameter_ratio"),
        "shape_mismatch_count": checkpoint_audit.get("shape_mismatch_count"),
        "mismatch_keys": checkpoint_audit.get("skipped_key_examples", []),
        "missing_keys_count": checkpoint_audit.get("missing_keys_count"),
        "unexpected_keys_count": checkpoint_audit.get("unexpected_keys_count"),
        "source_train_count": split_audit["source_train_count"],
        "support_count": split_audit["support_count"],
        "query_test_count": split_audit["query_test_count"],
        "support_per_class": split_audit["support_label_counts"],
        "source_label_counts": split_audit["source_label_counts"],
        "query_label_counts": split_audit["query_label_counts"],
        "split_overlap_all_zero": split_audit["all_overlap_zero"],
        "split_fingerprint": split_audit["fingerprint"],
        "loss_finite": bool(torch.isfinite(loss).item()),
        "no_nan_inf_input": bool(torch.isfinite(images).all().item()),
    }
    print("EEG_TOPOENCODER_CROSS_SUBJECT_DRYRUN_JSON=" + json.dumps(report, sort_keys=True))
    del model_image
    del model_text
    del model_clip
    return report



def _apply_runtime_config(config, method):
    protocol = "cross_subject"
    method_key = "etea_cgra" if method == "ETEA_CGRA" else "eeg_topoencoder"
    settings = ((config.get("experiments", {}) or {}).get(method_key, {}) or {}).get(protocol, {}) or {}
    for key in ["result_group", "pretrain", "checkpoint_loader_mode"]:
        if key in settings:
            config[key] = settings[key]
    if "wandb_name" in settings:
        config.setdefault("wandb_data", {})["name"] = settings["wandb_name"]
        config.setdefault("wandb_data", {})["desc"] = settings["wandb_name"]
    config.setdefault("etea_cgra", {})
    if method_key == "etea_cgra":
        config["etea_cgra"]["enabled"] = True
        config["etea_cgra"]["baseline_checkpoint"] = settings.get("baseline_checkpoint", "auto")
        config["etea_cgra"]["baseline_checkpoint_root"] = settings.get("baseline_checkpoint_root")
        config["etea_cgra"]["baseline_protocol"] = settings.get("baseline_protocol", protocol)
    else:
        config["etea_cgra"]["enabled"] = False
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-cfg", default="configs/SEED_IV.yaml")
    parser.add_argument("--method", choices=["EEG_TopoEncoder", "ETEA_CGRA"], default="EEG_TopoEncoder")
    parser.add_argument("--session", type=int, default=None)
    parser.add_argument("--test-subject", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    _validate_eeg_topoencoder_config(config)
    Trainer = _select_trainer(config)

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu_device_id"])
    utils.seed_everything(config["random_seed"])

    sessions = range(1, 4)
    subjects = range(1, 16)
    if args.session is not None or args.test_subject is not None:
        if args.session is None or args.test_subject is None:
            raise ValueError("--session and --test-subject must be provided together")
        sessions = [args.session]
        subjects = [args.test_subject]

    for session in sessions:
        for test_subject in subjects:
            print(f"Session = {session} Test_Subject = {test_subject}")
            source_train, support, query, manifest_path = _build_split(config, test_subject, session)
            split_audit = _audit_split(source_train, support, query, config["data"]["num_classes"])
            split_audit["manifest_path"] = manifest_path
            print("EEG_TOPOENCODER_CROSS_SUBJECT_SPLIT_JSON=" + json.dumps(split_audit, sort_keys=True))
            if not split_audit["all_overlap_zero"]:
                raise RuntimeError("split overlap audit failed")
            if split_audit["support_label_counts"] != [config["data"]["num_shot"]] * config["data"]["num_classes"]:
                raise RuntimeError("support per class audit failed")

            train_list = source_train + support
            if args.dry_run:
                _run_dry_run(config, train_list, query, split_audit, session, test_subject)
                continue

            model_image, model_text, model_clip, checkpoint_audit = _make_models(config)
            print("EEG_TOPOENCODER_CROSS_SUBJECT_TRAIN_CHECKPOINT_AUDIT_JSON=" + json.dumps(checkpoint_audit, sort_keys=True))
            start_epoch = config["solver"]["start_epoch"]

            parameters_image = filter(lambda p: p.requires_grad, model_image.parameters())
            parameters_image = sum([np.prod(p.size()) for p in parameters_image]) / 1_000_000
            print("model_image Trainable Parameters: %.3fM" % parameters_image)
            parameters_text = filter(lambda p: p.requires_grad, model_text.parameters())
            parameters_text = sum([np.prod(p.size()) for p in parameters_text]) / 1_000_000
            print("parameters_text Trainable Parameters: %.3fM" % parameters_text)
            print("Total Trainable Parameters: %.3fM" % (parameters_image + parameters_text))

            result_group = config.get("result_group", "SEED_EEGTopoEncoder_Transformer_FromScratch_CrossSubject_Clean")
            save_result_path = f"./SEED_train_result/{result_group}/session_{session}_test_subject_{test_subject}"
            trainer = Trainer(
                config=config,
                train_data_list=train_list,
                val_data_list=query,
                workers=config["data"]["workers"],
                model_text=model_text,
                model_image=model_image,
                start_epoch=start_epoch,
                save_result_path=save_result_path,
            )
            max_acc = trainer.training()
            del model_image
            del model_text
            del model_clip
            print(f"Clean SEED EEG-TopoEncoder Session = {session} Test_Subject = {test_subject} maxACC = {max_acc}")


if __name__ == "__main__":
    main()
