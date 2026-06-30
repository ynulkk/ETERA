import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from datasets import utils
from datasets.dataset import EEG_Dataset
from modules.ViViTEmotionNet import create_model
from modules.cfbm.checkpoint_loader import audit_checkpoint_load
from utils.pretrainer import Pretrainer


def _assert_eeg_topoencoder_config(config):
    net = config["network"]
    if net.get("feature") != "DE":
        raise ValueError(f"feature must be DE, got {net.get(feature)}")
    if int(net.get("image_channels", -1)) != 6:
        raise ValueError(f"image_channels must be 6, got {net.get(image_channels)}")
    if net.get("spectral_type") == "Legoformer":
        raise ValueError("EEG-TopoEncoder pretrain forbids Legoformer")


def _build_loader(config, file_list, batch_size, workers):
    ds = EEG_Dataset(
        images_path=file_list,
        image_height=config["network"]["image_height"],
        image_width=config["network"]["image_width"],
        num_classes=config["data"]["num_classes"],
        feature=config["network"]["feature"],
        map_type="SST",
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)


def _make_optimizer(config, model):
    name = config["solver"]["optim"]
    lr = config["solver"]["lr"]
    wd = config["solver"]["weight_decay"]
    if name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=config["solver"]["momentum"], weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name}")


def _run_dry_run(config, model, train_files, args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = model.to(device)
    model.train()

    loader = _build_loader(config, train_files, batch_size=args.dry_run_batch_size, workers=0)
    images, labels = next(iter(loader))
    input_shape = tuple(images.shape)
    if images.shape[2] != 6:
        raise RuntimeError(f"Expected EEG-TopoEncoder 6 channels, got {input_shape}")

    images = images.to(device).float()
    labels = labels.to(device)

    temporal_info = {"called": False, "input_shape": None, "output_shape": None}

    def temporal_hook(module, inputs, output):
        temporal_info["called"] = True
        temporal_info["input_shape"] = tuple(inputs[0].shape)
        temporal_info["output_shape"] = tuple(output.shape)

    hook = model.transformer.transformer_layers[2].register_forward_hook(temporal_hook)
    try:
        logits = model(images)
        loss = F.cross_entropy(logits, torch.argmax(labels, dim=1), reduction="mean")
        optimizer = _make_optimizer(config, model)
        optimizer.zero_grad()
        if args.dry_run_backward:
            loss.backward()
            optimizer.step()

        if not torch.isfinite(loss):
            raise FloatingPointError("dry-run loss is not finite")

        report = {
            "feature": config["network"]["feature"],
            "image_channels": config["network"]["image_channels"],
            "spectral_type": config["network"]["spectral_type"],
            "temporal_type": config["network"]["temporal_type"],
            "uses_psd": False,
            "uses_legoformer": False,
            "input_shape": input_shape,
            "labels_shape": tuple(labels.shape),
            "logits_shape": tuple(logits.shape),
            "loss": float(loss.detach().cpu().item()),
            "loss_finite": bool(torch.isfinite(loss).item()),
            "temporal_called": temporal_info["called"],
            "temporal_input_shape": temporal_info["input_shape"],
            "temporal_output_shape": temporal_info["output_shape"],
            "optimizer": config["solver"]["optim"],
            "checkpoint_mode": "from_scratch",
            "loaded_parameter_ratio": 0.0,
            "save_path": config.get("pretrain"),
            "no_nan_inf": bool(torch.isfinite(logits).all().item()),
        }
        print("EEG_TOPOENCODER_PRETRAIN_DRYRUN_REPORT_BEGIN")
        print(json.dumps(report, indent=2, sort_keys=True))
        print("EEG_TOPOENCODER_PRETRAIN_DRYRUN_REPORT_END")
    finally:
        hook.remove()


def _run_smoke(config, model, train_files, val_files, args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = model.to(device)
    model.train()

    train_loader = _build_loader(config, train_files, batch_size=args.smoke_batch_size, workers=0)
    val_loader = _build_loader(config, val_files, batch_size=args.smoke_batch_size, workers=0)
    optimizer = _make_optimizer(config, model)

    for epoch in range(args.smoke_epochs):
        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device).float()
            labels = labels.to(device)
            if images.shape[2] != 6:
                raise RuntimeError(f"Expected 6 channels in smoke train, got {tuple(images.shape)}")
            logits = model(images)
            loss = F.cross_entropy(logits, torch.argmax(labels, dim=1), reduction="mean")
            if not torch.isfinite(loss):
                raise FloatingPointError("smoke train loss is not finite")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step + 1 >= args.smoke_max_batches:
                break

    model.eval()
    with torch.no_grad():
        images, labels = next(iter(val_loader))
        images = images.to(device).float()
        labels = labels.to(device)
        logits = model(images)
        val_loss = F.cross_entropy(logits, torch.argmax(labels, dim=1), reduction="mean")
        if not torch.isfinite(val_loss):
            raise FloatingPointError("smoke val loss is not finite")

    out_dir = Path(args.smoke_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "model_best.pt"
    torch.save(
        {
            "epoch": args.smoke_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "tag": "eeg_topoencoder_pretrain_smoke",
        },
        ckpt_path,
    )

    report = {
        "smoke": True,
        "smoke_epochs": args.smoke_epochs,
        "smoke_max_batches": args.smoke_max_batches,
        "checkpoint_path": str(ckpt_path),
        "val_loss": float(val_loss.detach().cpu().item()),
        "no_nan_inf": bool(torch.isfinite(logits).all().item()),
    }
    print("EEG_TOPOENCODER_PRETRAIN_SMOKE_REPORT_BEGIN")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("EEG_TOPOENCODER_PRETRAIN_SMOKE_REPORT_END")


def _run_formal_pretrain(config, model, train_files, val_files):
    pretrain_save_path = os.path.dirname(config["pretrain"])
    if not pretrain_save_path:
        pretrain_save_path = f"./{config[data][dataset]}_pretrain_result/Temporal_{config[network][temporal_type]}"

    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_DISABLED", "true")
    device_ids = list(range(len(str(config["gpu_device_id"]).split(","))))
    if config.get("multi_gpu", False) and len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        model = model.cuda()


    pretrainer = Pretrainer(
        config=config,
        train_data_list=train_files,
        val_data_list=val_files,
        workers=config["data"]["workers"],
        model=model,
        loss_function=torch.nn.CrossEntropyLoss(reduction="sum"),
        save_result_path=pretrain_save_path,
    )
    best_metric = pretrainer.training()

    best_checkpoint = os.path.join(pretrainer.save_result_path, "model_best.pt")
    copied_checkpoint = None
    if os.path.isfile(best_checkpoint):
        os.makedirs(os.path.dirname(config["pretrain"]), exist_ok=True)
        shutil.copy2(best_checkpoint, config["pretrain"])
        copied_checkpoint = config["pretrain"]

    report = {
        "formal_pretrain": True,
        "save_result_path": pretrainer.save_result_path,
        "best_checkpoint_in_run_dir": best_checkpoint,
        "copied_checkpoint": copied_checkpoint,
        "best_metric": float(best_metric),
    }
    print("EEG_TOPOENCODER_PRETRAIN_FORMAL_REPORT_BEGIN")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("EEG_TOPOENCODER_PRETRAIN_FORMAL_REPORT_END")



def _apply_pretrain_config(config):
    settings = ((config.get("experiments", {}) or {}).get("pretrain", {}) or {})
    for key in ["result_group", "pretrain", "checkpoint_loader_mode"]:
        if key in settings:
            config[key] = settings[key]
    if "wandb_name" in settings:
        config.setdefault("wandb_data", {})["name"] = settings["wandb_name"]
        config.setdefault("wandb_data", {})["desc"] = settings["wandb_name"]
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/SEED_IV.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-batch-size", type=int, default=2)
    parser.add_argument("--dry-run-backward", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--smoke-max-batches", type=int, default=3)
    parser.add_argument("--smoke-batch-size", type=int, default=8)
    parser.add_argument("--smoke-output-dir", default="pretrain_smoke_outputs/EEG_TopoEncoder")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    config = _apply_pretrain_config(config)

    _assert_eeg_topoencoder_config(config)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu_device_id"])
    utils.seed_torch(config["random_seed"])

    train_npy_path_list, _ = utils.get_train_test_filets_path(
        config["data"]["datasets_path"],
        num_trial=config["data"]["num_trial"],
    )
    if len(train_npy_path_list) == 0:
        raise RuntimeError(f"No .npy files found under {config[data][datasets_path]}")

    train_path_list, val_path_list = train_test_split(
        train_npy_path_list,
        random_state=config["random_seed"],
        test_size=config["data"]["test_size"],
    )

    model = create_model(config)
    trainable = sum(np.prod(p.size()) for p in model.parameters() if p.requires_grad)
    audit, _ = audit_checkpoint_load(model, checkpoint_path=None, mode="from_scratch")

    print("EEG_TOPOENCODER_PRETRAIN_META_BEGIN")
    print(json.dumps({
        "checkpoint_mode": audit["checkpoint_loader_mode"],
        "feature": config["network"]["feature"],
        "image_channels": config["network"]["image_channels"],
        "loaded_parameter_ratio": audit["loaded_parameter_ratio"],
        "spectral_type": config["network"]["spectral_type"],
        "temporal_type": config["network"]["temporal_type"],
        "trainable_params": int(trainable),
        "uses_legoformer": False,
        "uses_psd": False,
    }, indent=2, sort_keys=True))
    print("EEG_TOPOENCODER_PRETRAIN_META_END")

    if args.dry_run:
        _run_dry_run(config, model, train_path_list, args)
        return

    if args.smoke:
        _run_smoke(config, model, train_path_list, val_path_list, args)
        return

    _run_formal_pretrain(config, model, train_path_list, val_path_list)


if __name__ == "__main__":
    main()
