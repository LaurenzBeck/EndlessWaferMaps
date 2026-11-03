"""# 🔊 Noise Sources

This module contains functions that create different 2d noise.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


def _import_torch():
    import torch

    return torch


_DEFAULT_DIAMOND_KERNEL = [[0.25, 0.0, 0.25], [0.0, 0.0, 0.0], [0.25, 0.0, 0.25]]
_DEFAULT_SQUARE_KERNEL = [[0.0, 0.25, 0.0], [0.25, 0.0, 0.25], [0.0, 0.25, 0.0]]


def pattern_noise(
    signal: np.array,
    strength: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """🔊 adds uniform noise to a pattern signal, where it is > 0.

    Args:
        signal (np.array): signal
        strength (float): determines the bounds for the noise U(-strength, 0)
        rng (np.random.Generator | int, optional): a random number generator or a seed for creating
            one with `np.random.default_rng`. Defaults to None.

    Returns:
        np.ndarray: noisy pattern signal
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    noise = rng.uniform(-strength, 0, size=signal.shape)

    return np.where(signal != 0, signal + noise, signal)


def beta_weibull(
    size: tuple[int, int],
    alpha: float = 1.0,
    beta: float = 1.0,
    weight_beta: float = 1.0,
    shape_weibull: float = 2.5,
    weight_weibull: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.array:
    """🐂 samples from a weighted combination of a Beta and a Weibull distribution.

    Args:
        size (tuple[int, int]): width and height of the noise image.
        alpha (float, optional): alpha parameter of the Beta distribution. Defaults to 1.0.
        beta (float, optional): beta parameter of the Beta distribution. Defaults to 1.0.
        weight_beta (float, optional): weight of the Beta distribution. Defaults to 1.0.
        shape_weibull (float, optional): shape parameter of the 1-parameter Weibull distribution. Defaults to 2.5.
        weight_weibull (float, optional): weight of the 1-parameter Weibull distribution. Defaults to 1.0.
        rng (np.random.Generator | int, optional): a random number generator or a seed for creating
            one with `np.random.default_rng`. Defaults to None.

    Returns:
        np.array: weighted beta-weibull noise
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    noise_w = rng.weibull(shape_weibull, size)
    noise_b = rng.beta(alpha, beta, size)
    return weight_weibull * noise_w + weight_beta * noise_b


def impulse_noise(
    size: tuple[int, int],
    fraction: float | None = None,
    fraction_log10: float | None = None,
    low: float = 0.0,
    high: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.array:
    """🔨 creates an image with sparse uniform(low, high) noise impulses.

    Args:
        size (tuple[int, int]): width and height of the noise image.
        fraction (float | None, optional): fraction of pixels that receive impulses.
            Kept for backward compatibility.
        fraction_log10 (float | None, optional): base-10 logarithm of the fraction
            of pixels that receive impulses, e.g. `-3 -> 0.001` and `-1 -> 0.1`.
        low (float, optional): lower boundary of the output interval. Defaults to 0.0.
        high (float, optional): upper boundary of the output interval. Defaults to 1.0.
        rng (np.random.Generator | int, optional): a random number generator or a seed for creating
            one with `np.random.default_rng`. Defaults to None.

    Returns:
        np.ndarray: impulse noise
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    x, y = size
    if fraction_log10 is not None:
        fraction = 10.0 ** float(fraction_log10)
    elif fraction is None:
        fraction = 0.01

    fraction = float(np.clip(fraction, 0.0, 1.0))
    num_impulses = min(x * y, math.ceil(fraction * x * y))

    noise = np.zeros(shape=size, dtype=np.float32)
    if num_impulses == 0:
        return noise

    flat_indices = rng.choice(x * y, size=num_impulses, replace=False)
    noise.flat[flat_indices] = rng.uniform(low, high, size=num_impulses)
    return noise


def pink_noise(size: tuple[int, int], roughness: float = 1.0, max_value: float = 1.0, rng: np.random.Generator | int = 42) -> np.array:
    """🪷 creates 2-dimensional scaled pink noise.

    Adapted from https://stackoverflow.com/questions/70085015/how-to-generate-2d-colored-noise

    Args:
        size (tuple[int, int]): width and height of the noise image.
        roughness (float, optional): determines how strong higher frequencies are present. Defaults to 1.0.
        max_value (float, optional): determines the scale of the noise (0, `max_value`). Defaults to 1.0.
        rng (np.random.Generator | int, optional): a random number generator or a seed for creating
            one with `np.random.default_rng`. Defaults to 42

    Returns:
        np.array: normalized pink_noise
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    white = rng.uniform(size=size)
    white_ft = np.fft.fftshift(np.fft.fft2(white))

    xsize = size[0] / 2
    ysize = size[1] / 2
    x, y = np.ogrid[-xsize:xsize, -ysize:ysize]
    freq = (x**2 + y**2) ** (1 / roughness)
    pink_ft = np.divide(white_ft, freq, out=np.zeros_like(white_ft), where=freq != 0)

    pink = np.fft.ifft2(np.fft.ifftshift(pink_ft)).real
    lo, hi = np.min(pink), np.max(pink)
    return max_value * ((pink - lo) / (hi - lo))


def diamond_square(
    shape: tuple[int, int],
    output_range: tuple[float, float] = (0.0, 1.0),
    roughness: float = 0.5,
    device: torch.device | str | None = None,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """💎 generates 2-dimensional fractal plasma noise using the diamond-square algorithm.

    The implementation was adapted from: https://github.com/anguelos/tormentor/blob/master/diamond_square/diamond_square.py

    Args:
        shape (tuple[int, int]): width and height of the noise image.
        output_range (tuple[float, float], optional): min and max values of the output. Defaults to (0.0, 1.0).
        roughness (float, optional): determines how strong higher frequencies are present. Defaults to 0.5.
        device (torch.device | str | None, optional): torch device to run this code on.
            If None, use the CPU. Defaults to None.
        rng (np.random.Generator | int, optional): a random number generator or a seed for creating
            one with `np.random.default_rng`. Defaults to None.

    Returns:
        torch.Tensor: 2D fractal plasma noise tensor
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    recursion_depth = max(math.log2(shape[0] - 1), math.log2(shape[1] - 1))
    recursion_depth = int(math.ceil(recursion_depth))

    if output_range is None:
        output_range = [0.0, 1.0]

    torch = _import_torch()

    with torch.no_grad():
        if device is None:
            device = "cpu"

        seed_img = torch.rand(1, 1, 3, 3).to(device)

        # ❇️ first diamond step
        rnd_range = 1.0 * roughness
        seed_img[:, :, 1, 1] = (
            0.25 * (seed_img[:, :, 0, 0] + seed_img[:, :, 2, 0] + seed_img[:, :, 0, 2] + seed_img[:, :, 2, 2])
            + torch.rand(1, device=device) * roughness
            - roughness / 2
        )
        # ⏹️ first square step
        seed_img[:, :, 0, 1] = (
            0.3333 * (seed_img[:, :, 0, 0] + seed_img[:, :, 0, 2] + seed_img[:, :, 1, 1])
            + torch.rand(1, device=device) * roughness
            - roughness / 2
        )
        seed_img[:, :, 1, 0] = (
            0.3333 * (seed_img[:, :, 0, 0] + seed_img[:, :, 2, 0] + seed_img[:, :, 1, 1])
            + torch.rand(1, device=device) * roughness
            - roughness / 2
        )
        seed_img[:, :, 2, 1] = (
            0.3333 * (seed_img[:, :, 2, 0] + seed_img[:, :, 2, 2] + seed_img[:, :, 1, 1])
            + torch.rand(1, device=device) * roughness
            - roughness / 2
        )
        seed_img[:, :, 1, 2] = (
            0.3333 * (seed_img[:, :, 0, 2] + seed_img[:, :, 2, 2] + seed_img[:, :, 1, 1])
            + torch.rand(1, device=device) * roughness
            - roughness / 2
        )
        rnd_range = rnd_range * roughness
        recursion_depth -= 1

        img = seed_img
        for _ in range(1, recursion_depth + 1):
            img = _one_diamond_one_square(img, rnd_range, rng)
            rnd_range *= roughness

        # 📏 scale image to `output_range`
        img_max = img.max(dim=2)[0].max(dim=2)[0].unsqueeze(dim=2).unsqueeze(dim=2)
        img_min = img.min(dim=2)[0].min(dim=2)[0].unsqueeze(dim=2).unsqueeze(dim=2)
        # * add epsilon of 0.00000001 to avoid division by 0
        img = (img - img_min) / (0.00000001 + img_max - img_min)
        img = img * (output_range[1] - output_range[0]) + output_range[0]

        if shape is not None:
            img = img[:, :, : shape[0], : shape[1]]

        return img[0, 0, :, :]


def _one_diamond_one_square(
    img: torch.Tensor,
    rnd_scale: float,
    rng: np.random.Generator,
    diamond_kernel: torch.Tensor | None = None,
    square_kernel: torch.Tensor | None = None,
) -> torch.Tensor:
    """❇️⏹️ doubles the image resolution by applying a single diamond square step.

    Recursive application of this method creates plasma fractals.
    Attention! The function is differentiable and gradients are computed as well.
    If this function is run in the usual sense, it is more efficient if it is run in a `no_grad()` context.

    Args:
        img (torch.Tensor): a 4D tensor where dimensions are Batch, Channel, Width, Height. Width and Height must both
            be 2^N+1 and Batch and Channels should in the usual case be 1.
        rnd_scale (float): a float in [0, 1] controlling the randomness created pixels get. In the usual case, it is
            halved at every application of this function.
        rng (np.random.Generator): random number generator.
        diamond_kernel (torch.Tensor | None, optional): kernel for the diamond step. Defaults to None.
        square_kernel (torch.Tensor | None, optional): kernel for the square step. Defaults to None.

    Returns:
        torch.Tensor: a tensor on the same device as `img` with the same channels and width, height of 2^(N+1)+1
    """
    torch = _import_torch()
    if diamond_kernel is None:
        diamond_kernel = torch.tensor(_DEFAULT_DIAMOND_KERNEL).unsqueeze(dim=0).unsqueeze(dim=0)
    if square_kernel is None:
        square_kernel = torch.tensor(_DEFAULT_SQUARE_KERNEL).unsqueeze(dim=0).unsqueeze(dim=0)

    batch_size, _, _, _ = img.size()
    step = 2
    new_img = torch.zeros([batch_size, 1, 2 * (img.shape[2] - 1) + 1, 2 * (img.shape[3] - 1) + 1], device=img.device)
    new_img[:, :, ::step, ::step] = img

    pad_compensate = torch.ones_like(new_img)
    pad_compensate[:, :, :, 0] = 1 / 0.75
    pad_compensate[:, :, :, -1] = 1 / 0.75
    pad_compensate[:, :, 0, :] = 1 / 0.75
    pad_compensate[:, :, -1, :] = 1 / 0.75

    rnd_img = torch.tensor(rng.uniform(size=list(new_img.size())), device=img.device).float() * rnd_scale

    # ❇️ diamond step
    diamond_regions = torch.nn.functional.conv2d(new_img, diamond_kernel.to(img.device), padding=1)
    diamond_centers = (diamond_regions > 0).float()
    new_img = new_img + (1 - rnd_scale) * diamond_regions * diamond_centers + diamond_centers * rnd_img

    # ⏹️ square step
    square_regions = torch.nn.functional.conv2d(new_img, square_kernel.to(img.device), padding=1) * pad_compensate
    square_centers = (square_regions > 0).float()

    return new_img + square_centers * rnd_img + (1 - rnd_scale) * square_centers * square_regions
