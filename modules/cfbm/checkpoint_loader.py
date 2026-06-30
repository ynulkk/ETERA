import torch


def _strip_prefix(state_dict, prefix="module."):
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {key[len(prefix):]: value for key, value in state_dict.items()}


def audit_checkpoint_load(model, checkpoint_path=None, mode="from_scratch"):
    target_state = model.state_dict()
    audit = {
        "checkpoint_loader_mode": mode,
        "source_key_count": 0,
        "target_key_count": len(target_state),
        "matched_key_count": 0,
        "loaded_parameter_ratio": 0.0,
        "missing_keys_count": len(target_state),
        "unexpected_keys_count": 0,
        "shape_mismatch_count": 0,
        "matched_key_examples": [],
        "skipped_key_examples": [],
        "notes": [],
    }
    if mode == "from_scratch" or not checkpoint_path:
        audit["notes"].append(
            "from_scratch: not using DE+PSD pretrain; loaded_parameter_ratio is 0 by design"
        )
        return audit, {}

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = checkpoint.get("model_state_dict", checkpoint)
    source_state = _strip_prefix(source_state)
    compatible = {}
    shape_mismatch = []
    unexpected = []
    for key, value in source_state.items():
        if key not in target_state:
            unexpected.append(key)
        elif target_state[key].shape == value.shape:
            compatible[key] = value
        else:
            shape_mismatch.append((key, tuple(value.shape), tuple(target_state[key].shape)))

    missing = [key for key in target_state if key not in compatible]
    source_params = sum(value.numel() for value in source_state.values() if torch.is_tensor(value))
    loaded_params = sum(value.numel() for value in compatible.values() if torch.is_tensor(value))
    audit.update(
        {
            "source_key_count": len(source_state),
            "matched_key_count": len(compatible),
            "loaded_parameter_ratio": float(loaded_params / source_params) if source_params else 0.0,
            "missing_keys_count": len(missing),
            "unexpected_keys_count": len(unexpected),
            "shape_mismatch_count": len(shape_mismatch),
            "matched_key_examples": list(compatible)[:8],
            "skipped_key_examples": [str(item) for item in (shape_mismatch[:8] or unexpected[:8])],
        }
    )
    return audit, compatible


def load_checkpoint_with_audit(model, checkpoint_path=None, mode="from_scratch", strict_eeg_topoencoder=True):
    audit, compatible = audit_checkpoint_load(model, checkpoint_path, mode)
    if mode == "from_scratch" or not checkpoint_path:
        return audit
    if strict_eeg_topoencoder and mode not in {"eeg_topoencoder_pretrain", "seediv_eeg_topoencoder_pretrain", "seed_cfbm_eeg_topoencoder_pretrain", "seediv_cfbm_eeg_topoencoder_pretrain"}:
        raise RuntimeError(f"Strict EEG-TopoEncoder run refuses checkpoint mode {mode}")
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    audit["missing_keys_count"] = len(missing)
    audit["unexpected_keys_count"] = len(unexpected)
    return audit
