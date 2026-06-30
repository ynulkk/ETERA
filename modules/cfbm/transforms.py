from .gaussian import gaussian_filter_2d


def _as_cfbm_config(config):
    if config is None:
        return {}
    if isinstance(config, dict) and "network" in config:
        return config.get("network", {}).get("eeg_topoencoder_cfbm", {}) or config.get("cfbm", {}) or {}
    return config or {}


def _validate_eeg_topoencoder_4d(images):
    if images.ndim != 5:
        raise ValueError(f"DE-only CFBM expects [B,T,6,H,W], got {tuple(images.shape)}")
    if images.shape[2] != 6:
        raise ValueError(f"DE-only CFBM refuses non-DE input: expected 6 channels, got {images.shape[2]}")


def apply_eeg_topoencoder_spatial_cfbm(images, kernel_size, sigma):
    _validate_eeg_topoencoder_4d(images)
    batch, frames, channels, height, width = images.shape

    flat = images.reshape(batch * frames * channels, 1, height, width)
    filtered = gaussian_filter_2d(flat, kernel_size=kernel_size, sigma=sigma)
    return filtered.reshape(batch, frames, channels, height, width)


def apply_eeg_topoencoder_cfbm_spatial_residual(
    images,
    kernel_size,
    sigma,
    alpha,
    residual_fusion=True,
    fusion_formula="convex",
):
    _validate_eeg_topoencoder_4d(images)
    alpha = float(alpha)
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"CFBM residual alpha must be in [0, 1], got {alpha}")
    if fusion_formula != "convex":
        raise ValueError(f"Unsupported CFBM fusion_formula: {fusion_formula}")

    filtered = apply_eeg_topoencoder_spatial_cfbm(images, kernel_size=kernel_size, sigma=sigma)
    if residual_fusion:
        mixed = images * (1.0 - alpha) + filtered * alpha
    else:
        mixed = filtered
    if mixed.shape != images.shape:
        raise RuntimeError(f"CFBM residual changed shape from {tuple(images.shape)} to {tuple(mixed.shape)}")
    if mixed.dtype != images.dtype:
        mixed = mixed.to(dtype=images.dtype)
    if mixed.device != images.device:
        mixed = mixed.to(device=images.device)
    if not mixed.isfinite().all():
        raise FloatingPointError("CFBM residual fusion produced NaN or Inf")
    return mixed


def maybe_apply_eeg_topoencoder_cfbm(images, config):
    cfg = _as_cfbm_config(config)
    if not cfg.get("enabled", False):
        if images.ndim != 5 or images.shape[2] != 6:
            raise ValueError(f"Raw DE-only path expects [B,T,6,H,W], got {tuple(images.shape)}")
        return images

    mode = cfg.get("mode", "spatial_gaussian")
    if mode not in {"spatial_gaussian", "spatial_eeg_topoencoder"}:
        raise ValueError(f"Unsupported DE-only CFBM mode: {mode}")
    gaussian_cfg = cfg.get("gaussian", {}) or {}
    kernel_size = gaussian_cfg.get("kernel_size", cfg.get("kernel_size", 3))
    sigma = gaussian_cfg.get("sigma", cfg.get("sigma", 0.5))
    if cfg.get("residual_fusion", False):
        return apply_eeg_topoencoder_cfbm_spatial_residual(
            images,
            kernel_size=kernel_size,
            sigma=sigma,
            alpha=cfg.get("alpha", 1.0),
            residual_fusion=True,
            fusion_formula=cfg.get("fusion_formula", "convex"),
        )
    return apply_eeg_topoencoder_spatial_cfbm(
        images,
        kernel_size=kernel_size,
        sigma=sigma,
    )
