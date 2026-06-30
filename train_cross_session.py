import argparse
import hashlib
import json
import os
from pathlib import Path

import clip
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader

from datasets import utils
from datasets.dataset import EEG_Dataset
from modules.ViViTEmotionNet import create_model
from modules.cfbm.checkpoint_loader import audit_checkpoint_load, load_checkpoint_with_audit
from modules.cfbm.transforms import apply_eeg_topoencoder_spatial_cfbm, maybe_apply_eeg_topoencoder_cfbm
from utils import Text_Prompt
from utils.trainer_eeg_topoencoder import Trainer as DefaultTrainer


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


def _write_manifest(path, payload):
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _class_counts(files, num_classes):
    counts = {str(label): 0 for label in range(num_classes)}
    for path in files:
        label = int(Path(path).stem.split("_")[-1])
        counts[str(label)] += 1
    return counts


def _assert_no_overlap(train_files, support_files, test_files):
    checks = {
        "train_vs_support": set(train_files) & set(support_files),
        "train_vs_test": set(train_files) & set(test_files),
        "support_vs_test": set(support_files) & set(test_files),
    }
    for name, overlap in checks.items():
        if overlap:
            raise RuntimeError(f"{name} has overlap, examples: {sorted(overlap)[:3]}")


def _strip_prefix(state_dict, prefix):
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {key[len(prefix):]: value for key, value in state_dict.items()}


def _load_pretrained_encoder(model_image, checkpoint_path, mode="eeg_topoencoder_pretrain"):
    return load_checkpoint_with_audit(
        model_image.model,
        checkpoint_path=checkpoint_path,
        mode=mode,
        strict_eeg_topoencoder=True,
    )


def _checkpoint_mode(config):
    mode = config.get("checkpoint_loader_mode")
    if mode:
        return mode
    return "eeg_topoencoder_pretrain" if config.get("isPretrain", False) else "from_scratch"


def _fold_fingerprint(manifest):
    payload = json.dumps(
        {
            "train": manifest["train_session_files"],
            "support": manifest["support_files"],
            "test": manifest["query_files"],
            "seed": manifest["seed"],
            "subject": manifest["subject"],
            "train_session": manifest["train_session"],
            "test_session": manifest["test_session"],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _assert_eeg_topoencoder_config(config):
    network = config["network"]
    if network.get("feature") != "DE":
        raise ValueError(f"Strict EEG-TopoEncoder requires network.feature == DE, got {network.get('feature')}")
    if int(network.get("image_channels")) != 6:
        raise ValueError(f"Strict EEG-TopoEncoder requires image_channels == 6, got {network.get('image_channels')}")
    if network.get("spectral_type") == "Legoformer":
        raise ValueError("Strict EEG-TopoEncoder refuses Legoformer because it splits channels as DE/PSD halves")


def _eeg_topoencoder_cfbm_cfg(config):
    return config.get("network", {}).get("eeg_topoencoder_cfbm", {}) or config.get("cfbm", {}) or {}


def _eeg_topoencoder_cfbm_audit(config):
    cfg = _eeg_topoencoder_cfbm_cfg(config)
    gaussian_cfg = cfg.get("gaussian", {}) or {}
    return {
        "dataset": config["data"]["dataset"],
        "num_classes": config["data"]["num_classes"],
        "feature": config["network"].get("feature"),
        "image_channels": config["network"].get("image_channels"),
        "spectral_type": config["network"].get("spectral_type"),
        "temporal_type": config["network"].get("temporal_type"),
        "pretrain": config.get("pretrain"),
        "isPretrain": config.get("isPretrain"),
        "isResume": config.get("isResume"),
        "checkpoint_loader_mode": _checkpoint_mode(config),
        "result_group": config.get("result_group"),
        "cfbm_enabled": cfg.get("enabled", False),
        "cfbm_mode": cfg.get("mode"),
        "cfbm_residual_fusion": cfg.get("residual_fusion", False),
        "cfbm_fusion_formula": cfg.get("fusion_formula", "convex"),
        "cfbm_alpha": cfg.get("alpha", 1.0),
        "cfbm_kernel_size": gaussian_cfg.get("kernel_size", cfg.get("kernel_size")),
        "cfbm_sigma": gaussian_cfg.get("sigma", cfg.get("sigma")),
    }


def _print_config_audit(config):
    print("EEG_TOPOENCODER_CFBM_CONFIG_AUDIT_BEGIN")
    print(json.dumps(_eeg_topoencoder_cfbm_audit(config), indent=2, sort_keys=True))
    print("EEG_TOPOENCODER_CFBM_CONFIG_AUDIT_END")


def _compute_class_logits(image_features, text_features, num_text_aug, num_classes, logit_scale):
    image_features = F.normalize(image_features, dim=-1)
    similarity = float(logit_scale) * (image_features @ text_features.T)
    similarity = similarity.view(image_features.shape[0], num_text_aug, num_classes)
    return similarity.mean(dim=1)


def _run_dry_run(config, train_with_support, test_files, manifest, args):
    _assert_eeg_topoencoder_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu_dry_run else "cpu")
    model = create_model(config).to(device)
    model.eval()

    checkpoint_mode = _checkpoint_mode(config)
    checkpoint_path = config.get("pretrain") if checkpoint_mode != "from_scratch" else None
    if checkpoint_mode == "from_scratch":
        audit = load_checkpoint_with_audit(model, mode="from_scratch")
    elif checkpoint_mode in {"eeg_topoencoder_pretrain", "seediv_eeg_topoencoder_pretrain", "seed_cfbm_eeg_topoencoder_pretrain", "seediv_cfbm_eeg_topoencoder_pretrain"}:
        audit = load_checkpoint_with_audit(
            model,
            checkpoint_path=checkpoint_path,
            mode=checkpoint_mode,
            strict_eeg_topoencoder=True,
        )
    else:
        raise RuntimeError(f"Strict EEG-TopoEncoder dry-run refuses checkpoint mode {checkpoint_mode}")

    temporal_info = {"called": False, "input_shape": None, "output_shape": None}

    def temporal_hook(module, inputs, output):
        temporal_info["called"] = True
        temporal_info["input_shape"] = tuple(inputs[0].shape)
        temporal_info["output_shape"] = tuple(output.shape)

    hook = model.transformer.transformer_layers[2].register_forward_hook(temporal_hook)
    try:
        dataset = EEG_Dataset(
            images_path=train_with_support,
            image_height=config["network"]["image_height"],
            image_width=config["network"]["image_width"],
            num_classes=config["data"]["num_classes"],
            feature=config["network"]["feature"],
            map_type="SST",
        )
        loader = DataLoader(
            dataset,
            batch_size=args.dry_run_batch_size,
            shuffle=False,
            num_workers=0,
        )
        images, labels = next(iter(loader))
        input_shape = tuple(images.shape)
        if images.shape[2] != 6:
            raise RuntimeError(f"PSD present or non-DE input detected: input shape {input_shape}")
        images = images.to(device).float()
        labels = labels.to(device)
        cfg = _eeg_topoencoder_cfbm_cfg(config)
        cfbm_enabled = bool(cfg.get("enabled", False))
        if cfbm_enabled:
            gaussian_cfg = cfg.get("gaussian", {}) or {}
            kernel_size = gaussian_cfg.get("kernel_size", cfg.get("kernel_size", 3))
            sigma = gaussian_cfg.get("sigma", cfg.get("sigma", 0.5))
            filtered_cfbm = apply_eeg_topoencoder_spatial_cfbm(images, kernel_size=kernel_size, sigma=sigma)
        else:
            kernel_size = None
            sigma = None
            filtered_cfbm = images
        mixed = maybe_apply_eeg_topoencoder_cfbm(images, config)
        output_shape = tuple(mixed.shape)
        if not torch.isfinite(images).all():
            raise FloatingPointError("Dry-run raw input has NaN or Inf")
        if cfbm_enabled and not torch.isfinite(filtered_cfbm).all():
            raise FloatingPointError("Dry-run filtered CFBM has NaN or Inf")
        if not torch.isfinite(mixed).all():
            raise FloatingPointError("Dry-run mixed CFBM has NaN or Inf")

        model_clip, _ = clip.load("./clip_weights/ViT-B-16.pt", device=device)
        model_clip.eval()
        model_image = ImageCLIP(model).to(device)
        model_text = TextCLIP(model_clip).to(device)
        model_image.eval()
        model_text.eval()
        classes, num_text_aug, _ = Text_Prompt.eeg_text_prompt(config["data"]["classes_names"])
        with torch.no_grad():
            text_features = model_text(classes.to(device))
            text_features = F.normalize(text_features, dim=-1)
            image_features = model_image(mixed)
            logits = _compute_class_logits(
                image_features,
                text_features,
                num_text_aug,
                config["data"]["num_classes"],
                config["solver"].get("logit_scale", 1.0),
            )
        loss = F.cross_entropy(logits, torch.argmax(labels, dim=1), reduction="mean")
        if not torch.isfinite(loss):
            raise FloatingPointError("Dry-run loss is not finite")
        if not temporal_info["called"]:
            raise RuntimeError("Dry-run temporal encoder was not called")
        if output_shape != input_shape:
            raise RuntimeError(f"Dry-run model input shape wrong: expected {input_shape}, got {output_shape}")
        if tuple(logits.shape) != (input_shape[0], config["data"]["num_classes"]):
            raise RuntimeError(f"Dry-run logits shape wrong: got {tuple(logits.shape)}")

        report = {
            "version": config["wandb_data"]["name"],
            "dataset": config["data"]["dataset"],
            "num_classes": config["data"]["num_classes"],
            "raw_input_shape": input_shape,
            "filtered_cfbm_shape": tuple(filtered_cfbm.shape),
            "mixed_output_shape": output_shape,
            "model_input_shape": output_shape,
            "psd_present": False,
            "legoformer_used": config["network"].get("spectral_type") == "Legoformer",
            "cfbm_enabled": cfg.get("enabled", False),
            "cfbm_mode": cfg.get("mode", "spatial_gaussian"),
            "kernel_size": kernel_size,
            "sigma": sigma,
            "residual_fusion": cfg.get("residual_fusion", False),
            "fusion_formula": cfg.get("fusion_formula", "convex"),
            "alpha": cfg.get("alpha", 1.0),
            "temporal_called": temporal_info["called"],
            "temporal_input_shape": temporal_info["input_shape"],
            "temporal_output_shape": temporal_info["output_shape"],
            "eeg_text_matching_retained": True,
            "image_features_shape": tuple(image_features.shape),
            "text_features_shape": tuple(text_features.shape),
            "logits_shape": tuple(logits.shape),
            "loss_finite": bool(torch.isfinite(loss).item()),
            "loss": float(loss.detach().cpu().item()),
            "checkpoint_mode": audit["checkpoint_loader_mode"],
            "loaded_parameter_ratio": audit["loaded_parameter_ratio"],
            "checkpoint_audit": audit,
            "split_fingerprint": _fold_fingerprint(manifest),
            "overlap": 0,
            "no_nan_inf": True,
            "train_sample_count": manifest["counts"]["train_total"],
            "test_sample_count": manifest["counts"]["test"],
            "support_label_counts": manifest["counts"]["support_by_class"],
            "test_label_counts": manifest["counts"]["test_by_class"],
        }
        print("EEG_TOPOENCODER_CFBM_DRY_RUN_REPORT_BEGIN")
        print(json.dumps(report, indent=2, sort_keys=True))
        print("EEG_TOPOENCODER_CFBM_DRY_RUN_REPORT_END")
    finally:
        hook.remove()



def _apply_runtime_config(config, method):
    protocol = "cross_session"
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
    parser.add_argument("--config", default="configs/SEED_IV.yaml")
    parser.add_argument("--method", choices=["EEG_TopoEncoder", "ETEA_CGRA"], default="EEG_TopoEncoder")
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--train-session", type=int, required=True)
    parser.add_argument("--test-session", type=int, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pretrain-checkpoint", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-batch-size", type=int, default=2)
    parser.add_argument("--cpu-dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    config = _apply_runtime_config(config, args.method)
    if args.pretrain_checkpoint:
        config["pretrain"] = args.pretrain_checkpoint
    if args.wandb_name:
        config["wandb_data"]["name"] = args.wandb_name
        config["wandb_data"]["desc"] = args.wandb_name
    if args.method == "ETEA_CGRA":
        from utils.trainer_etea_cgra import ETEACGRATrainer
        Trainer = ETEACGRATrainer
    else:
        Trainer = DefaultTrainer

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(config["gpu_device_id"]))
    utils.seed_torch(config["random_seed"])
    _print_config_audit(config)

    train_files, test_candidates = utils.get_subject_cross_session_train_test_filets_path(
        data_path=config["data"]["datasets_path"],
        subjectID=args.subject,
        num_trial=config["data"]["num_trial"],
        train_session=args.train_session,
        test_session=args.test_session,
    )
    support_files, test_files = utils.cross_subject_n_shot(
        test_candidates,
        num_shot=config["data"]["num_shot"],
        num_classes=config["data"]["num_classes"],
        seed=config["random_seed"],
        manifest_path=args.manifest,
        manifest_metadata={
            "fold_type": "cross_session_original_source_split",
            "subject": args.subject,
            "train_session": args.train_session,
            "test_session": args.test_session,
        },
        dataset=config["data"]["dataset"],
        session=args.test_session,
        test_subject=args.subject,
    )
    _assert_no_overlap(train_files, support_files, test_files)
    train_with_support = train_files + support_files
    split_manifest = {}
    if Path(args.manifest).exists():
        with Path(args.manifest).open("r") as f:
            split_manifest = json.load(f)

    manifest = {
        "dataset": config["data"]["dataset"],
        "fold_type": "cross_session_original_source_split",
        "subject": args.subject,
        "train_session": args.train_session,
        "test_session": args.test_session,
        "random_seed": config["random_seed"],
        "pretrain_checkpoint": config.get("pretrain"),
        "counts": {
            "train_session": len(train_files),
            "support": len(support_files),
            "train_total": len(train_with_support),
            "test_candidates": len(test_candidates),
            "test": len(test_files),
            "support_by_class": _class_counts(support_files, config["data"]["num_classes"]),
            "test_by_class": _class_counts(test_files, config["data"]["num_classes"]),
        },
        "shot": config["data"]["num_shot"],
        "seed": config["random_seed"],
        "path_sorting": split_manifest.get("path_sorting", utils.PATH_SORTING),
        "generated_at": split_manifest.get("generated_at"),
        "train_session_files": sorted(train_files),
        "support": sorted(support_files),
        "test": sorted(test_files),
        "support_files": sorted(support_files),
        "query_files": sorted(test_files),
    }
    _write_manifest(args.manifest, manifest)
    print("Original source cross-session manifest written:", args.manifest)
    print("Counts:", manifest["counts"])

    if args.dry_run:
        _run_dry_run(config, train_with_support, test_files, manifest, args)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_clip, _ = clip.load("./clip_weights/ViT-B-16.pt", device=device)
    model_image = ImageCLIP(create_model(config))
    model_text = TextCLIP(model_clip)

    _assert_eeg_topoencoder_config(config)
    checkpoint_mode = _checkpoint_mode(config)
    if checkpoint_mode == "from_scratch":
        audit = load_checkpoint_with_audit(model_image.model, mode="from_scratch")
    elif checkpoint_mode in {"eeg_topoencoder_pretrain", "seediv_eeg_topoencoder_pretrain", "seed_cfbm_eeg_topoencoder_pretrain", "seediv_cfbm_eeg_topoencoder_pretrain"}:
        audit = _load_pretrained_encoder(model_image, config["pretrain"], mode=checkpoint_mode)
    else:
        raise RuntimeError(f"Strict EEG-TopoEncoder training refuses checkpoint mode {checkpoint_mode}")
    print("Checkpoint audit:", json.dumps(audit, indent=2, sort_keys=True))

    model_image = model_image.cuda()
    model_text = model_text.cuda()
    for param in model_text.parameters():
        param.requires_grad = False

    result_group = config.get("result_group", "Original_Cross_session_From_SEED_Pretrain")
    save_result_path = (
        f"./SEED_IV_train_result/{result_group}/"
        f"train_session{args.train_session}_test_session{args.test_session}_subject_{args.subject}"
    )
    trainer = Trainer(
        config=config,
        train_data_list=train_with_support,
        val_data_list=test_files,
        workers=config["data"]["workers"],
        model_text=model_text,
        model_image=model_image,
        start_epoch=config["solver"]["start_epoch"],
        save_result_path=save_result_path,
    )
    max_acc = trainer.training()
    print(
        f"Original source cross-session EEG-TopoEncoder CFBM result: train_session={args.train_session} "
        f"test_session={args.test_session} subject={args.subject} maxACC={max_acc}"
    )


if __name__ == "__main__":
    main()
