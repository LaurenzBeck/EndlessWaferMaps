"""# 🟰 Functions

This module contains 1-dimensional functions.
"""

from typing import Callable

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d


def generate_spline_function(
    num_points: int,
    min_val: float,
    max_val: float,
    x_range: tuple[float, float] = (0.0, 1.0),
    sigma: float = 1.0,
    clip: bool = False,  # noqa: FBT001, FBT002
    rng: np.random.Generator | int = 42,
) -> Callable[[float], float]:
    """🐍 generate a random spline function.

    For this, `num_points-2` sampled points between `x_range` and the range bounds themselves
    will be filtered with a gaussian filter with standard deviation `sigma`.
    These points will then be used to fit a `scipy.interpolate.CubicSpline` model.

    Args:
        num_points (int): number of points to sample in the domain
        min_val (float): minimum value of the y interval
        max_val (float): maximum value of the y interval
        x_range (tuple[float, float], optional): x interval. Defaults to (0.0, 1.0)
        sigma (float, optional): standard deviation of the Gaussian filter. Defaults to 1.0
        clip (bool, optional): whether to clip the results of the spline model to `[min_val, max_val]`. Defaults to False
        rng (np.random.Generator | int, optional): a random number generator or a seed for creating
            one with `np.random.default_rng`. Defaults to 42

    Returns:
        f (Callable[[float], float]): Spline function that maps x to y
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    # Sample points in the domain, fixing the extremes
    x_random = rng.uniform(x_range[0], x_range[1], num_points - 2)
    x = np.concatenate([[x_range[0]], np.sort(x_random), [x_range[1]]])
    y = rng.uniform(min_val, max_val, num_points)

    # smooth the y values using a Gaussian filter
    y_smooth = gaussian_filter1d(y, sigma=sigma)

    # fit a cubic spline model
    # * bc_type=((1, 0), (1, 0)) sets the boundary conditions -> first derivative at extremes should be 0
    cs = CubicSpline(x, y_smooth, bc_type=((1, 0), (1, 0)))

    return lambda x: np.clip(cs(x), a_min=min_val, a_max=max_val) if clip else cs(x)
