import torch
import torch.nn.functional as F


def build_gaussian_kernel2d(kernel_size, sigma, device=None, dtype=None):
    if isinstance(kernel_size, (tuple, list)):
        if len(kernel_size) != 2 or kernel_size[0] != kernel_size[1]:
            raise ValueError("kernel_size must be an odd int or square tuple")
        kernel_size = int(kernel_size[0])
    kernel_size = int(kernel_size)
    sigma = float(sigma)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    coords = torch.arange(kernel_size, device=device, dtype=dtype or torch.float32)
    coords = coords - (kernel_size - 1) / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(torch.finfo(kernel.dtype).eps)
    return kernel.view(1, 1, kernel_size, kernel_size)


def gaussian_filter_2d(x, kernel_size, sigma):
    if x.ndim != 4:
        raise ValueError(f"gaussian_filter_2d expects [N,1,H,W], got {tuple(x.shape)}")
    if x.shape[1] != 1:
        raise ValueError(f"gaussian_filter_2d expects a single channel, got {x.shape[1]}")

    original_dtype = x.dtype
    kernel = build_gaussian_kernel2d(kernel_size, sigma, device=x.device, dtype=x.dtype)
    padding = int(kernel_size) // 2
    filtered = F.conv2d(x, kernel, padding=padding)
    norm = F.conv2d(torch.ones_like(x), kernel, padding=padding)
    filtered = filtered / norm.clamp_min(torch.finfo(x.dtype).eps)
    if not torch.isfinite(filtered).all():
        raise FloatingPointError("Gaussian filtering produced NaN or Inf")
    return filtered.to(dtype=original_dtype)
