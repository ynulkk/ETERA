"""ETEA-CGRA trainer: baseline-anchored residual-only training.

ETEA_CGRA-A warm-starts the EEG backbone from a canonical baseline checkpoint,
freezes that backbone, and trains only the confidence-gated EEG residual.
"""

import glob
import os
import re
import time

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from modules.cfbm.transforms import maybe_apply_eeg_topoencoder_cfbm
from modules.etea_cgra.confidence_gated_residual_adapter import ConfidenceGatedEEGResidualAdapter
from utils import Text_Prompt
from utils.trainer_eeg_topoencoder import (
    Trainer as BaseTrainer,
    _move_to_cuda_float,
    _batch_size,
)


class ETEACGRATrainer(BaseTrainer):
    """Baseline warm-start + frozen backbone + residual-only trainer."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config", args[0] if args else None)
        if config is None:
            raise ValueError("ETEACGRATrainer requires config")

        self.etea_cgra_cfg = config.get("etea_cgra", {}) or {}
        self.etea_cgra_enabled = bool(self.etea_cgra_cfg.get("enabled", False))
        self.etea_cgra = None
        self.baseline_checkpoint_path = None
        self._etea_cgra_bootstrap_optimizer = True
        self._etea_cgra_logged_logits_shape = False

        super().__init__(*args, **kwargs)

        if self.etea_cgra_enabled:
            self.etea_cgra = self._build_etea_cgra_module().cuda()
            self.baseline_checkpoint_path = self._resolve_baseline_checkpoint()
            self._load_baseline_backbone(self.baseline_checkpoint_path)
            self._freeze_model_image()
            self.optimizer = self._init_residual_only_optimizer(self.config)
            self.lr_scheduler = self._init_lr_scheduler(self.config)
            self._load_etea_cgra_checkpoint_if_requested()
            self._print_etea_cgra_audit()
        self._etea_cgra_bootstrap_optimizer = False

    def _build_etea_cgra_module(self):
        classes, num_text_aug, _ = Text_Prompt.eeg_text_prompt(self.classes_names)
        del classes
        network_cfg = self.config.get("network", {}) or {}
        cfg = self.etea_cgra_cfg
        feature_dim = int(cfg.get("feature_dim", network_cfg.get("hidden_dims", 512)))
        common_kwargs = dict(
            feature_dim=feature_dim,
            num_classes=self.num_classes,
            num_text_aug=num_text_aug,
            alpha=float(cfg.get("alpha", 0.02)),
            hidden_ratio=float(cfg.get("hidden_ratio", 0.25)),
            dropout=float(cfg.get("dropout", 0.0)),
            gate_type=str(cfg.get("gate_type", "scalar")),
            detach_confidence=bool(cfg.get("detach_confidence", True)),
            normalize_features=bool(cfg.get("normalize_features", True)),
            residual_zero_init=bool(cfg.get("residual_zero_init", True)),
            gate_init_bias=float(cfg.get("gate_init_bias", -2.0)),
            debug_shapes=bool(cfg.get("debug_shapes", False)),
        )
        mode = str(cfg.get("mode", "") or "").lower()
        if mode in {"residual_only", "confidence_gate_only"}:
            gate_input_mode = cfg.get("gate_input_mode")
            if gate_input_mode is None:
                gate_input_mode = "confidence" if mode == "confidence_gate_only" else "full"
            print("ETEACGRA ETEA_CGRA mode={}".format(mode))
            return ETEA_CGRAAblationEEGAdapter(
                **common_kwargs,
                mode=mode,
                gate_input_mode=str(gate_input_mode),
                residual_uses_confidence=bool(cfg.get("residual_uses_confidence", False)),
                context_delta_normalize=bool(cfg.get("context_delta_normalize", True)),
                context_mode=str(cfg.get("context_mode", "soft")),
            )
        return ConfidenceGatedEEGResidualAdapter(**common_kwargs)

    def _init_optimizer(self, config, model_image):
        # BaseTrainer calls this before ETEA_CGRA has loaded and frozen the backbone.
        # Build a temporary optimizer, then replace it with residual-only optimizer.
        if getattr(self, "_etea_cgra_bootstrap_optimizer", False):
            return optim.AdamW(
                [{"params": model_image.parameters(), "lr": config["solver"]["lr"]}],
                betas=(0.9, 0.98),
                lr=config["solver"]["lr"],
                eps=1e-8,
                weight_decay=config["solver"]["weight_decay"],
            )
        return self._init_residual_only_optimizer(config)

    def _baseline_protocol(self):
        return str(self.etea_cgra_cfg.get("baseline_protocol", "seediv_cross_session"))

    def _parse_fold_from_save_path(self):
        protocol = self._baseline_protocol()
        if protocol in {"seediv_cross_subject", "seed_cross_subject"}:
            match = re.search(
                r"session_(\d+)_test_subject_(\d+)",
                self.save_result_path or "",
            )
            if not match:
                raise RuntimeError(
                    "ETEACGRA cannot infer cross-subject fold from save_result_path: {}".format(
                        self.save_result_path
                    )
                )
            session, subject = match.groups()
            return {
                "protocol": protocol,
                "session": int(session),
                "subject": int(subject),
            }

        match = re.search(
            r"train_session(\d+)_test_session(\d+)_subject_(\d+)",
            self.save_result_path or "",
        )
        if not match:
            raise RuntimeError(
                "ETEACGRA cannot infer fold from save_result_path: {}".format(
                    self.save_result_path
                )
            )
        train_session, test_session, subject = match.groups()
        return {
            "protocol": protocol,
            "subject": int(subject),
            "train_session": int(train_session),
            "test_session": int(test_session),
        }

    def _resolve_baseline_checkpoint(self):
        cfg = self.etea_cgra_cfg
        checkpoint = cfg.get("baseline_checkpoint", "auto")
        if checkpoint and checkpoint != "auto":
            if not os.path.exists(checkpoint):
                raise FileNotFoundError("Configured baseline checkpoint not found: {}".format(checkpoint))
            print("Resolved baseline checkpoint:", checkpoint)
            return checkpoint

        fold = self._parse_fold_from_save_path()
        protocol = fold["protocol"]
        root = cfg.get(
            "baseline_checkpoint_root",
            "SEED_IV_train_result/EEGTopoEncoder_CFBM_Strict_FromScratch_DryRun",
        )
        if protocol in {"seediv_cross_subject", "seed_cross_subject"}:
            pattern = os.path.join(
                root,
                "session_{}_test_subject_{}".format(fold["session"], fold["subject"]),
                "*",
                "model_best.pt",
            )
        else:
            pattern = os.path.join(
                root,
                "train_session{}_test_session{}_subject_{}".format(
                    fold["train_session"],
                    fold["test_session"],
                    fold["subject"],
                ),
                "*",
                "model_best.pt",
            )
        candidates = sorted(glob.glob(pattern))
        if not candidates:
            raise FileNotFoundError("No baseline checkpoint matched pattern: {}".format(pattern))

        for path in reversed(candidates):
            try:
                ckpt = torch.load(path, map_location="cpu")
            except Exception as exc:  # pragma: no cover - diagnostic path
                print("Skip unreadable baseline checkpoint {}: {}".format(path, exc))
                continue
            if isinstance(ckpt, dict) and "ViViTEmotion_state_dict" in ckpt:
                print("ETEACGRA baseline_protocol:", protocol)
                print("Resolved baseline checkpoint:", path)
                return path
            print("Skip baseline checkpoint without ViViTEmotion_state_dict:", path)

        raise RuntimeError("No usable baseline checkpoint found under: {}".format(pattern))

    def _load_baseline_backbone(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("ViViTEmotion_state_dict") if isinstance(ckpt, dict) else None
        if state_dict is None:
            raise RuntimeError("Baseline checkpoint missing ViViTEmotion_state_dict: {}".format(checkpoint_path))
        state_dict = self._normalize_baseline_state_dict(state_dict)
        missing, unexpected = self.model_image.load_state_dict(state_dict, strict=True)
        print(
            "Loaded baseline ViViTEmotion_state_dict strict=True missing={} unexpected={}".format(
                missing,
                unexpected,
            )
        )
        print("Baseline checkpoint epoch:", ckpt.get("epoch", "NA"))

    def _normalize_baseline_state_dict(self, state_dict):
        target_keys = list(self.model_image.state_dict().keys())
        source_keys = list(state_dict.keys())
        if not target_keys or not source_keys:
            return state_dict

        target_has_module = target_keys[0].startswith("module.")
        source_has_module = source_keys[0].startswith("module.")
        if source_has_module and not target_has_module:
            print("Normalized baseline state_dict: stripped module. prefix")
            return {
                key[len("module."):] if key.startswith("module.") else key: value
                for key, value in state_dict.items()
            }
        if target_has_module and not source_has_module:
            print("Normalized baseline state_dict: added module. prefix")
            return {
                key if key.startswith("module.") else "module." + key: value
                for key, value in state_dict.items()
            }
        return state_dict

    def _freeze_model_image(self):
        if not bool(self.etea_cgra_cfg.get("freeze_backbone", True)):
            print("ETEACGRA freeze_backbone=false; model_image remains trainable")
            return
        for param in self.model_image.parameters():
            param.requires_grad = False
        self.model_image.eval()

    def _count_trainable_params(self, module):
        return sum(param.numel() for param in module.parameters() if param.requires_grad)

    def _init_residual_only_optimizer(self, config):
        if self.etea_cgra is None:
            raise RuntimeError("ETEACGRA residual module is not initialized")
        params = [param for param in self.etea_cgra.parameters() if param.requires_grad]
        if not params:
            raise RuntimeError("ETEACGRA residual module has no trainable parameters")
        param_groups = [{"params": params, "lr": config["solver"]["lr"]}]

        if config["solver"]["optim"] == "Adam":
            optimizer = optim.Adam(
                param_groups,
                lr=config["solver"]["lr"],
                betas=(0.9, 0.98),
                eps=1e-8,
                weight_decay=0.2,
            )
            print("Adam")
        elif config["solver"]["optim"] == "SGD":
            optimizer = optim.SGD(
                param_groups,
                config["solver"]["lr"],
                momentum=config["solver"]["momentum"],
                weight_decay=config["solver"]["weight_decay"],
            )
            print("SGD")
        elif config["solver"]["optim"] == "AdamW":
            optimizer = optim.AdamW(
                param_groups,
                betas=(0.9, 0.98),
                lr=config["solver"]["lr"],
                eps=1e-8,
                weight_decay=config["solver"]["weight_decay"],
            )
            print("AdamW")
        else:
            raise ValueError("Unknown optimizer: {}".format(config["solver"]["optim"]))

        print("optimizer_param_groups = residual_only")
        print("optimizer_param_group_count = {}".format(len(optimizer.param_groups)))
        return optimizer

    def _load_etea_cgra_checkpoint_if_requested(self):
        resume_path = self.etea_cgra_cfg.get("resume")
        if not resume_path and self.config.get("isResume", False):
            resume_path = self.config.get("resume")
        if not resume_path:
            return
        if not os.path.exists(resume_path):
            print("ETEACGRA resume checkpoint not found, skip:", resume_path)
            return
        checkpoint = torch.load(resume_path, map_location="cpu")
        state_dict = checkpoint.get("etea_cgra_state_dict")
        if state_dict is None:
            print("ETEACGRA state_dict missing in checkpoint, skip:", resume_path)
            return
        missing, unexpected = self.etea_cgra.load_state_dict(state_dict, strict=False)
        print("ETEACGRA resume loaded: missing={}, unexpected={}".format(missing, unexpected))

    def _print_etea_cgra_audit(self):
        loss_cfg = self.etea_cgra_cfg.get("loss", {}) or {}
        print(
            "ETEA-CGRA config: alpha={} gate_type={} detach_confidence={} "
            "ce1_weight={} lambda_gate={} lambda_xshift={} residual_zero_init={} "
            "gate_init_bias={} freeze_backbone={} train_residual_only={}".format(
                self.etea_cgra_cfg.get("alpha", 0.02),
                self.etea_cgra_cfg.get("gate_type", "scalar"),
                self.etea_cgra_cfg.get("detach_confidence", True),
                loss_cfg.get("ce1_weight", 1.0),
                loss_cfg.get("lambda_gate", 0.0),
                loss_cfg.get("lambda_xshift", 0.005),
                self.etea_cgra_cfg.get("residual_zero_init", True),
                self.etea_cgra_cfg.get("gate_init_bias", -2.0),
                self.etea_cgra_cfg.get("freeze_backbone", True),
                self.etea_cgra_cfg.get("train_residual_only", True),
            )
        )
        print("model_image_trainable_params = {}".format(self._count_trainable_params(self.model_image)))
        print("etea_cgra_trainable_params = {}".format(self._count_trainable_params(self.etea_cgra)))
        print("ETEACGRA mode = {}".format(self.etea_cgra_cfg.get("mode", "etea_cgra")))
        print("ETEACGRA context_mode = {}".format(self.etea_cgra_cfg.get("context_mode", "soft")))
        print("ETEACGRATrainer enabled: baseline_warmstart_frozen_backbone_residual_only")

    def _save_training_state(self, epoch, filename, max_acc=None):
        payload = {
            "epoch": epoch,
            "baseline_checkpoint_path": self.baseline_checkpoint_path,
            "ViViTEmotion_state_dict": self.model_image.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "etea_cgra_enabled": self.etea_cgra_enabled,
            "freeze_backbone": bool(self.etea_cgra_cfg.get("freeze_backbone", True)),
            "train_residual_only": bool(self.etea_cgra_cfg.get("train_residual_only", True)),
            "mode": self.etea_cgra_cfg.get("mode", "etea_cgra"),
        }
        if max_acc is not None:
            payload["maxACC"] = max_acc
        if self.etea_cgra is not None:
            payload["etea_cgra_state_dict"] = self.etea_cgra.state_dict()
        torch.save(payload, filename)

    def _save_best(self, epoch, max_acc):
        self._save_training_state(epoch, os.path.join(self.save_result_path, "model_best.pt"), max_acc=max_acc)

    def _save_epoch(self, epoch, max_acc):
        self._save_training_state(epoch, os.path.join(self.save_result_path, "model_epoch_save.pt"), max_acc=max_acc)

    def _compute_etea_cgra_etea_cgra_logits(self, image_features, text_features, num_text_aug):
        logits0 = self._compute_class_logits(image_features, text_features, num_text_aug)
        output = self.etea_cgra(image_features, text_features, logits0)
        logits1 = self._compute_class_logits(output["x"], text_features, num_text_aug)
        if not self._etea_cgra_logged_logits_shape:
            print(
                "ETEACGRA logits_shape logits0={} logits1={} num_classes={}".format(
                    tuple(logits0.shape),
                    tuple(logits1.shape),
                    self.num_classes,
                )
            )
            self._etea_cgra_logged_logits_shape = True
        return logits1, [logits0, logits1], output

    def _weighted_etea_cgra_loss(self, logits_list, labels_idx, output):
        loss_cfg = self.etea_cgra_cfg.get("loss", {}) or {}
        ce1_weight = float(loss_cfg.get("ce1_weight", 1.0))
        lambda_gate = float(loss_cfg.get("lambda_gate", 0.0))
        lambda_xshift = float(loss_cfg.get("lambda_xshift", 0.005))

        loss = ce1_weight * self.loss_function(logits_list[-1], labels_idx)
        if output is not None:
            loss = loss + lambda_gate * output["gate_mean"]
            loss = loss + lambda_xshift * output["x_shift_norm"]
        return loss

    def _validate(self, epoch, classes, val_loader, num_text_aug):
        self.model_image.eval()
        self.model_text.eval()
        self.etea_cgra.eval()

        num = 0
        corr_0 = 0
        corr_1 = 0
        gate_sum = 0.0
        xshift_sum = 0.0

        with torch.no_grad():
            text_features = self._encode_text_features(classes)
            for images, labels in tqdm(val_loader):
                batch_size = _batch_size(images)
                labels = torch.argmax(labels, dim=-1).cuda()
                images = _move_to_cuda_float(images)
                images = maybe_apply_eeg_topoencoder_cfbm(images, self.config)
                image_features = self.model_image(images)

                logits1, logits_list, output = self._compute_etea_cgra_etea_cgra_logits(
                    image_features,
                    text_features,
                    num_text_aug,
                )
                logits0 = logits_list[0]
                _, preds0 = logits0.topk(1, dim=-1)
                _, preds1 = logits1.topk(1, dim=-1)
                labels_view = labels.view(-1, 1)
                corr_0 += (preds0 == labels_view).sum().item()
                corr_1 += (preds1 == labels_view).sum().item()
                num += batch_size
                gate_sum += float(output["gate_mean"].item()) * batch_size
                xshift_sum += float(output["x_shift_norm"].item()) * batch_size

                for pred, label in zip(preds1, labels_view):
                    self.confusion_matrix[label.item()][pred.item()] += 1

        logits0_acc = float(corr_0) / num * 100
        logits1_acc = float(corr_1) / num * 100
        gate_mean = gate_sum / num
        x_shift_norm = xshift_sum / num
        print(
            "[{}/{}][Testing]: logits0_acc: {} logits1_acc: {} "
            "logits1_minus_logits0: {} val_acc: {} gate_mean: {} x_shift_norm: {}".format(
                epoch + 1,
                self.num_epochs,
                logits0_acc,
                logits1_acc,
                logits1_acc - logits0_acc,
                logits1_acc,
                gate_mean,
                x_shift_norm,
            )
        )
        return logits1_acc

    def training(self):
        if not self.etea_cgra_enabled:
            return super().training()

        self.loss_function = torch.nn.CrossEntropyLoss(reduction="sum")

        train_loader = self._get_data_loader(self.train_data_list, self.config["solver"]["train_batch_size"])
        val_loader = self._get_data_loader(self.val_data_list, self.config["solver"]["val_batch_size"])

        classes, num_text_aug, _ = Text_Prompt.eeg_text_prompt(self.classes_names)

        if self.save_result_path is None:
            self.save_result_path = "../train_result"
        self.save_result_path = os.path.join(
            self.save_result_path,
            time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time())),
        ).replace("\\", "/")
        if not os.path.exists(self.save_result_path):
            os.makedirs(self.save_result_path)

        best_prec1 = 0.0
        self.model_text.eval()
        text_features = self._encode_text_features(classes)
        num_early_patience = 0
        for epoch in range(self.start_epoch, self.num_epochs):
            print("Epoch: {}/{}".format(epoch + 1, self.num_epochs))

            self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)
            train_total_loss, corr_0, corr_1, num_total = 0.0, 0, 0, 0
            gate_sum, xshift_sum = 0.0, 0.0

            self.model_image.eval()
            self.model_text.eval()
            self.etea_cgra.train()
            for images, labels in tqdm(train_loader):
                batch_size = _batch_size(images)
                images = _move_to_cuda_float(images)
                images = maybe_apply_eeg_topoencoder_cfbm(images, self.config)
                labels = labels.cuda()
                labels_idx = torch.argmax(labels, dim=1)

                with torch.no_grad():
                    image_embedding = self.model_image(images)
                logits1, logits_list, output = self._compute_etea_cgra_etea_cgra_logits(
                    image_embedding,
                    text_features.detach(),
                    num_text_aug,
                )
                loss = self._weighted_etea_cgra_loss(logits_list, labels_idx, output)

                train_total_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                _, predictions0 = torch.max(logits_list[0].data, dim=1)
                _, predictions1 = torch.max(logits1.data, dim=1)
                corr_0 += (predictions0 == labels_idx).sum().item()
                corr_1 += (predictions1 == labels_idx).sum().item()
                num_total += batch_size
                gate_sum += float(output["gate_mean"].detach().item()) * batch_size
                xshift_sum += float(output["x_shift_norm"].detach().item()) * batch_size

            avg_train_loss = float(train_total_loss) / num_total
            train_acc0 = float(corr_0) / num_total * 100
            train_acc1 = float(corr_1) / num_total * 100
            print(
                "[{}/{}][Training]: train_loss: {} logits0_train_acc: {} "
                "logits1_train_acc: {} logits1_minus_logits0: {} gate_mean: {} x_shift_norm: {}".format(
                    epoch + 1,
                    self.num_epochs,
                    avg_train_loss,
                    train_acc0,
                    train_acc1,
                    train_acc1 - train_acc0,
                    gate_sum / num_total,
                    xshift_sum / num_total,
                )
            )

            val_total_acc_top1 = self._validate(epoch, classes, val_loader, num_text_aug)
            if val_total_acc_top1 > best_prec1:
                best_prec1 = max(val_total_acc_top1, best_prec1)

                precision, recall, f1 = self._calc_precision_recall_f1()
                accuracy = self._calc_accuracy()
                print("Confusion Matrix: ")
                print(self.confusion_matrix)
                print("Precision: " + str(precision))
                print("Recall: " + str(recall))
                print("F1: " + str(f1))
                print("mAccuracy: " + str(accuracy))
                print("mPrecision: " + str(np.mean(precision)))
                print("mRecall: " + str(np.mean(recall)))
                print("mF1: " + str(np.mean(f1)))
                num_early_patience = 0
                self._save_best(epoch, best_prec1)
            else:
                if (epoch + 1) % self.config["epoch_save_freq"] == 0:
                    self._save_epoch(epoch, best_prec1)
                num_early_patience += 1
                if self.config["solver"]["is_early_patience"] and num_early_patience >= self.config["solver"]["early_patience"]:
                    print(f"Early stopping triggered! max val_acc = {best_prec1}")
                    break

        print(f"Training end. max val_acc = {best_prec1}")
        return best_prec1
