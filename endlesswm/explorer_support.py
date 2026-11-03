"""Helpers for local marimo explorer notebooks."""

from __future__ import annotations

import inspect
import re
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.ndimage import distance_transform_edt

from .geometry import full_ellipse, linear_gradient, linear_radial_gradient
from .noise import beta_weibull, impulse_noise, pink_noise
from .patterns import pattern_generators, pattern_params
from .transforms import gaussian_filter
from .trees import background_params
from .utils import affine
from .visualizations import compact_heatmap, plot_mask

DEFAULT_SHAPE = (52, 52)
REFERENCE_PATTERN = "center_spot"


def midpoint_parameter_values(parameter_bounds: dict[str, Any]) -> dict[str, Any]:
    """Resolve midpoint defaults for one parameter block."""
    values: dict[str, Any] = {}
    for name, bound in parameter_bounds.items():
        if name.endswith("_r"):
            values[name[:-2]] = float(bound[0] + bound[1]) / 2.0
        elif name.endswith("_c"):
            values[name[:-2]] = bound[0]
        else:
            values[name] = bound
    return values


def widget_values_to_kwargs(
    widget_values: dict[str, Any],
    key_map: dict[str, str] | None = None,
    skip_parameters: set[str] | None = None,
) -> dict[str, Any]:
    """Convert widget values keyed by display names into callable kwargs."""
    kwargs: dict[str, Any] = {}
    skip = skip_parameters or set()
    for display_name, widget_value in widget_values.items():
        param_name = key_map[display_name] if key_map is not None else display_name
        if param_name in skip:
            continue
        if param_name.endswith("_r"):
            kwargs[param_name[:-2]] = float(widget_value)
        elif param_name.endswith("_c"):
            kwargs[param_name[:-2]] = widget_value
        else:
            kwargs[param_name] = widget_value
    return kwargs


def _call_with_optional_rng(func: Callable[..., Any], *args: Any, seed: int, **kwargs: Any) -> Any:
    if "rng" in inspect.signature(func).parameters:
        kwargs["rng"] = np.random.default_rng(int(seed))
    return func(*args, **kwargs)


def apply_wafer_mask(array: np.ndarray) -> np.ndarray:
    """Apply the wafer ellipse mask to a 2D array."""
    mask = full_ellipse(array.shape)
    return np.where(mask == 0.0, np.nan, array)


def fill_masked_array_with_nearest_valid_values(
    array: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Fill off-wafer and NaN regions with the nearest valid wafer value."""
    values = np.asarray(array, dtype=np.float32)
    if mask is None:
        mask = full_ellipse(values.shape)

    invalid_mask = (np.asarray(mask) == 0.0) | np.isnan(values)
    if not np.any(invalid_mask):
        return np.array(values, copy=True)
    if np.all(invalid_mask):
        return np.zeros_like(values, dtype=np.float32)

    nearest_indices = distance_transform_edt(invalid_mask, return_distances=False, return_indices=True)
    filled = np.array(values, copy=True)
    filled[invalid_mask] = values[tuple(nearest_indices)][invalid_mask]
    return filled


def _build_reference_background(seed: int, shape: tuple[int, int]) -> np.ndarray:
    bg = beta_weibull(size=shape, rng=np.random.default_rng(seed), **midpoint_parameter_values(background_params["beta_weibull"]))
    bg = bg + pink_noise(size=shape, rng=np.random.default_rng(seed + 1), **midpoint_parameter_values(background_params["pink_noise"]))
    bg = gaussian_filter(bg, **midpoint_parameter_values(background_params["gaussian_filter"]))
    bg = bg + impulse_noise(
        size=shape,
        rng=np.random.default_rng(seed + 3),
        **midpoint_parameter_values(background_params["impulse_noise"]),
    )
    bg = bg * linear_gradient(shape=shape, **midpoint_parameter_values(background_params["linear_gradient"]))
    bg = bg * linear_radial_gradient(shape=shape, **midpoint_parameter_values(background_params["linear_radial_gradient"]))
    bg = affine(bg, **midpoint_parameter_values(background_params["affine"]))
    return apply_wafer_mask(np.asarray(bg, dtype=np.float32))


def build_pattern_generator_kwargs(
    pattern_name: str,
    parameter_values: dict[str, Any],
    shape: tuple[int, int] | None = None,
    skip_parameters: set[str] | None = None,
) -> dict[str, Any]:
    """Build a kwargs dict for one pattern generator."""
    kwargs: dict[str, Any] = {}
    skip = skip_parameters or set()

    for param_name, param_spec in pattern_params[pattern_name].items():
        if param_name in skip:
            continue
        display_name = param_name[:-2] if param_name.endswith(("_r", "_c")) else param_name
        if param_name.endswith("_r"):
            kwargs[param_name[:-2]] = float(parameter_values[display_name])
        elif param_name.endswith("_c"):
            kwargs[param_name[:-2]] = parameter_values[display_name]
        else:
            kwargs[param_name] = param_spec

    generator = pattern_generators[pattern_name]
    if shape is not None and "shape" in inspect.signature(generator).parameters:
        kwargs.setdefault("shape", shape)

    return kwargs


def build_pattern_corruption_kwargs(
    corruption_function: Callable[..., Any],
    parameter_values: dict[str, Any],
    shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Build a kwargs dict for one pattern corruption call."""
    kwargs = dict(parameter_values)
    if shape is not None and "shape" in inspect.signature(corruption_function).parameters:
        kwargs.setdefault("shape", shape)
    return kwargs


def generate_pattern_signal(
    pattern_name: str, parameter_values: dict[str, Any], seed: int, shape: tuple[int, int] = DEFAULT_SHAPE
) -> np.ndarray:
    """Generate one pattern signal from a pattern node."""
    signal = np.zeros(shape, dtype=np.float32)
    generator = pattern_generators[pattern_name]
    kwargs = dict(parameter_values)
    if "shape" in inspect.signature(generator).parameters:
        kwargs.setdefault("shape", shape)
    output = _call_with_optional_rng(generator, signal, seed=seed, **kwargs)
    return np.asarray(output, dtype=np.float32)


def build_reference_inputs(
    seed: int,
    shape: tuple[int, int] = DEFAULT_SHAPE,
    reference_pattern: str = REFERENCE_PATTERN,
) -> dict[str, np.ndarray]:
    """Build the reference inputs used by the node explorer."""
    rng = np.random.default_rng(int(seed))
    gaussian_noise = apply_wafer_mask(rng.normal(loc=0.0, scale=1.0, size=shape).astype(np.float32))
    background = _build_reference_background(seed + 10, shape)
    pattern_defaults = {
        k: v for k, v in midpoint_parameter_values(pattern_params[reference_pattern]).items() if k not in {"polarity", "value"}
    }
    pattern_signal = apply_wafer_mask(generate_pattern_signal(reference_pattern, pattern_defaults, seed + 20, shape))
    full_sample = apply_wafer_mask(np.nan_to_num(background, nan=0.0) + np.nan_to_num(pattern_signal, nan=0.0))
    return {
        "gaussian noise": gaussian_noise,
        "background map": background,
        "pattern signal": pattern_signal,
        "full wafer map sample": full_sample,
    }


def build_default_gif_path_from_dataset_path(dataset_path: str | Path) -> Path:
    """Build the default GIF path next to a dataset path."""
    resolved = Path(dataset_path)
    return resolved.with_name(f"{resolved.stem}_samples.gif")


def build_explorer_gif_path(output_path: str | Path, default_output_path: str | Path | None = None) -> Path:
    """Resolve a GIF output path while preserving the chosen directory."""
    raw = re.sub(r"[\r\n]+", "", str(output_path)).strip()
    if raw:
        candidate = Path(raw)
    elif default_output_path is not None:
        candidate = Path(default_output_path)
    else:
        raise ValueError("A default GIF path is required when no output path is provided.")

    if candidate.suffix.lower() != ".gif":
        candidate = candidate.with_suffix(".gif")

    cleaned_name = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate.name.strip()).strip("._") or "wafer_map_samples"
    if not cleaned_name.lower().endswith(".gif"):
        cleaned_name = f"{cleaned_name}.gif"
    return candidate.with_name(cleaned_name)


def open_path_in_file_browser(path: str | Path) -> None:
    """Reveal a file in the platform file browser."""
    resolved = Path(path).resolve()
    if sys.platform.startswith("win"):
        subprocess.run(["explorer", f"/select,{resolved}"], check=False)
    elif sys.platform == "darwin":
        subprocess.run(["open", "-R", str(resolved)], check=False)
    else:
        subprocess.run(["xdg-open", str(resolved.parent)], check=False)


def render_sample_grid_frame(
    samples: list[np.ndarray],
    sample_view: str,
    rows: int,
    columns: int,
):
    """Render one grid frame for an animated sample GIF."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from PIL import Image

    figure = Figure(figsize=(columns * 2.35, rows * 2.35))
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, columns, squeeze=False)

    for idx, axis in enumerate(axes.flat):
        if idx >= len(samples):
            axis.set_axis_off()
            continue

        arr = np.asarray(samples[idx])
        if sample_view == "mask annotations":
            plot_mask(arr, create_fig=False, ax=axis, legend=False)
        else:
            if sample_view == "pattern":
                arr = np.where(arr == 0, np.nan, arr)
            compact_heatmap(apply_wafer_mask(np.asarray(arr, dtype=np.float32)), ax=axis)

    figure.tight_layout()

    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=140)
    buffer.seek(0)
    frame = Image.open(buffer).convert("RGBA").copy()
    buffer.close()
    return frame


def save_sample_grid_gif(frames: list[Any], output_path: str | Path, frame_duration_ms: int) -> Path:
    """Write rendered sample grid frames to an animated GIF file."""
    from PIL import Image

    if not frames:
        raise ValueError("Expected at least one rendered frame when saving a GIF.")

    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    palette_frames = [frame.convert("P", palette=Image.ADAPTIVE) for frame in frames]
    palette_frames[0].save(
        resolved,
        save_all=True,
        append_images=palette_frames[1:],
        duration=int(frame_duration_ms),
        loop=0,
        optimize=False,
        disposal=2,
    )
    return resolved
