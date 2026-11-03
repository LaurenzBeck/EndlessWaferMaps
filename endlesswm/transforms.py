"""# ➡️ Transformations

This module contains functions that transform arrays into other arrays with the same shape.
"""

from typing import Any, Callable

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import AffineTransform, resize, warp
from skimage.transform import swirl as swirl_skimage


def to_dict(sample, key: str) -> dict[str, Any]:  # noqa: ANN001
    """📖 puts the input in a dictionary using the `key` as an identifier."""
    return {key: sample}


def apply_radially(sample: np.array, func: Callable[[float], float]) -> np.ndarray:
    """🛞 applies the weighting `func` radially to `sample`.

    For this, the normalized distance to the center, which is capped at 1,
    will be used to look up the positional weight using `func`.

    Args:
        sample (np.array): input sample
        func (Callable[[float], float]): function that maps the clipped distance
            to the center (between 0 and 1) to a multiplicative weight

    Returns:
        np.ndarray: radially weighted sample
    """
    positional_weight = np.ones(sample.shape)
    center_x = sample.shape[0] // 2
    center_y = sample.shape[1] // 2

    for x in range(sample.shape[0]):
        for y in range(sample.shape[1]):
            norm_distance = np.clip(np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) / max(center_x, center_y), 0.0, 1.0)
            positional_weight[x, y] = func(norm_distance)

    return sample * positional_weight


def local_pixel_shuffle(image: np.array, fraction: float = 0.1, radius: float = 0.075, rng: np.random.Generator | int = 42) -> np.array:
    """👾 shuffles a `fraction` of pixels inside the patterns with pixels within a `radius`.

    Args:
        image (np.array): input image
        fraction (float, optional): fraction of the pixels to be shuffled. Defaults to 0.1.
        radius (float, optional): maximum radius of the shuffle distance as a fraction of the image dim. Defaults to 0.075.
        rng (np.random.Generator | int, optional): a random number generator or a seed for creating
            one with `np.random.default_rng`. Defaults to 42

    Returns:
        np.array: shuffled image
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    # Identify edge pixels
    pattern_pixels = np.where(image > 0)

    radius = max(1, int(radius * min(*image.shape)))

    for i, j in zip(pattern_pixels[0], pattern_pixels[1], strict=False):
        if rng.random() < fraction:
            # Randomly select a pixel within the corruption radius
            dx, dy = rng.integers(-radius, radius + 1, size=2)
            x, y = i + dx, j + dy
            x, y = np.clip(x, 0, image.shape[0] - 1), np.clip(y, 0, image.shape[1] - 1)
            # Swap pixel values
            image[i, j], image[x, y] = image[x, y], image[i, j]

    return image


def random_shrink_pad(image: np.array, scale_range: tuple[float, float] = (0.5, 0.9), rng: np.random.Generator | int = 42) -> np.array:
    """⬇️ resizes an image to a random fraction from `scale_range` and padding the result with zeros to keep the original shape.

    Args:
        image (np.array): input image
        scale_range (tuple[float, float], optional): range from which the scale fraction will be sampled from. Defaults to (0.5, 0.9).
        rng (np.random.Generator | int, optional): a random number generator or a seed for creating
            one with `np.random.default_rng`. Defaults to 42

    Returns:
        np.array: transformed image
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    height, width = image.shape

    # Randomly select the crop scale
    scale = rng.uniform(scale_range[0], scale_range[1])

    # Compute the crop dimensions
    crop_width = int(width * scale)
    crop_height = int(height * scale)

    image = resize(image, (crop_height, crop_width))

    diff_w = width - crop_width
    diff_h = height - crop_height

    return np.pad(
        image,
        (
            (diff_h // 2, diff_h // 2 if diff_h % 2 == 0 else diff_h // 2 + 1),
            (diff_w // 2, diff_w // 2 if diff_w % 2 == 0 else diff_w // 2 + 1),
        ),
        "constant",
        constant_values=0,
    )


def random_grow_trim(image: np.array, scale_range: tuple[float, float] = (1.1, 1.5), rng: np.random.Generator | int = 42) -> np.array:
    """⬆️ resizes an image to a random multiple from `scale_range` and trims the result to keep the original shape.

    Args:
        image (np.array): input image
        scale_range (tuple[float, float], optional): range from which the scale fraction will be sampled from. Defaults to (1.1, 1.5).
        rng (np.random.Generator | int, optional): a random number generator or a seed for creating
            one with `np.random.default_rng`. Defaults to 42

    Returns:
        np.array: transformed image
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    height, width = image.shape

    # Randomly select the grow scale
    scale = rng.uniform(scale_range[0], scale_range[1])

    # Compute the grow dimensions
    grow_width = int(width * scale)
    grow_height = int(height * scale)

    diff_w = grow_width - width
    diff_h = grow_height - height

    # if the difference is too small, skip the transform
    if diff_w == 0 or diff_h == 0:
        return image

    image = resize(image, (grow_height, grow_width))

    x_start = diff_w // 2
    y_start = diff_h // 2

    x_end = x_start if diff_w % 2 == 0 else x_start + 1

    y_end = y_start if diff_h % 2 == 0 else y_start + 1

    return image[y_start:-y_end, x_start:-x_end]


def slant(image: np.array, scale: float, rotation: float, shear: float) -> np.array:
    """🔄 applies a perspective transformation to a 2D numpy array, simulating a slight slanting angle.

    Args:
        image (np.array): input image
        scale (float): scale factor
        rotation (float): rotation angle in radians
        shear (float): shear angle in radians

    Returns:
        np.array: transformed image
    """
    tform = AffineTransform(scale=scale, rotation=rotation, shear=shear)
    return warp(image, inverse_map=tform.inverse)


def smooth_normalize(pattern: np.array, sigma: float = 1.0) -> np.array:
    """🎨 applies a gaussian kernel to a given pattern to achieve a smooth transition effect.

    Args:
        pattern (np.array): pattern to be smoothed
        sigma (float): standard deviation for the gaussian kernel

    Returns:
        np.array: pattern with applied gaussian kernel
    """
    smoothed_pattern = gaussian_filter(pattern, sigma=sigma)

    if (smoothed_pattern.max() - smoothed_pattern.min()).item() == 0.0:
        return smoothed_pattern
    return (smoothed_pattern - smoothed_pattern.min()) / (smoothed_pattern.max() - smoothed_pattern.min())


def swirl(image: np.array, strength: float) -> np.array:
    """🌀 applies `skimage.transform.swirl` to an image.

    Args:
        image (np.array): input image
        strength (float): the amount of swirling applied

    Returns:
        np.array: transformed image
    """
    return swirl_skimage(image, strength=strength, mode="constant", cval=0, preserve_range=True)
