"""# 🖼️ Visualizations"""

# ruff: noqa: F403, F405

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, Callable, Literal

import IPython
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib import animation, cm
from matplotlib.figure import Figure
from scipy.fft import fft2, fftshift
from skimage.transform import radon

from .patterns import classes
from .utils import *

if TYPE_CHECKING:
    import torch

    from endlesswm.io.datasets import H5WM

matplotlib.style.use("ggplot")
sns.set_theme()

warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    module="skimage.transform.radon_transform",
)


def heatmap(wafermap: np.ndarray, title: str | None = None, create_fig: bool = True, **kwargs: dict[str, Any]) -> Figure | None:  # noqa: FBT001, FBT002
    """🖼️ creates a heatmap visualization of a wafer map.

    Args:
        wafermap (np.ndarray): 2d wafer map.
        title (str | None, optional): optional title for the heatmap. Defaults to None.
        create_fig (bool, optional): whether this function should create a new figure object. Defaults to True.
        kwargs (dict[str, Any]): additional keyword arguments are passed to `seaborn.heatmap`.

    Returns:
        Figure | None: `create_fig` ? figure : None
    """
    if create_fig:
        fig = plt.figure()

    cmap = kwargs.pop("cmap", "jet")

    ax = sns.heatmap(wafermap.T, cmap=cmap, cbar=True, **kwargs)
    ax.grid(visible=True)
    ax.invert_yaxis()
    if title:
        ax.set_title(title)

    return fig if create_fig else None


def plot_mask(
    mask: np.ndarray,
    create_fig: bool = True,
    legend: bool = False,
    alpha: float = 0.75,
    **kwargs: dict[str, Any],
) -> Figure | None:
    """🎭 creates a visualization of wafer pattern masks.

    Args:
        mask (np.ndarray): 3d (c x w x h) wafer mask.
        create_fig (bool, optional): whether this function should create a new figure object. Defaults to True.
        legend (bool, optional): whether to include a legend. Defaults to False.
        alpha (float, optional): alpha value of the masks. Defaults to 0.75.
        kwargs (dict[str, Any]): additional keyword arguments are passed to `plt.imshow`.

    Returns:
        Figure | None: `create_fig` ? figure : None
    """
    if create_fig:
        fig, ax = plt.subplots()
    else:
        ax = kwargs.pop("ax")

    cmap = plt.get_cmap("tab20", len(classes))
    colors = [(*cmap(i / len(classes))[:3], alpha) for i in range(len(classes))]  # Add alpha to RGB

    # Plot each class mask with its specific color and transparency
    for i in range(len(classes)):
        class_mask = mask[i]
        colored_mask = np.zeros((*class_mask.shape, 4))
        for j in range(3):
            colored_mask[..., j] = class_mask * colors[i][j]
        colored_mask[..., 3] = class_mask * colors[i][3]
        ax.imshow(
            np.transpose(colored_mask, (1, 0, 2)),
            aspect="auto",
            **kwargs,
        )  # Swap the first two dimensions have the same orientation we use in the other figs

    ax.grid(visible=True)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

    if legend:
        legend_elements = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[i], markeredgewidth=0, markersize=10, label=class_name)
            for i, class_name in enumerate(classes)
        ]
        ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5))

    return fig if create_fig else None


def compact_heatmap(wafermap: np.ndarray, title: str | None = None, create_fig: bool = False, **kwargs: dict[str, Any]) -> Figure | None:  # noqa: FBT001, FBT002
    """🖼️ creates a compact heatmap visualization of a wafer map without ticks and cbar.

    Args:
        wafermap (np.ndarray): 2d wafer map.
        title (str | None, optional): optional title for the heatmap. Defaults to None.
        create_fig (bool, optional): whether this function should create a new figure object. Defaults to False.
        kwargs (dict[str, Any]): additional keyword arguments are passed to `seaborn.heatmap`.

    Returns:
        Figure | None: `create_fig` ? figure : None
    """
    if create_fig:
        fig = plt.figure()

    cmap = kwargs.pop("cmap", "jet")

    ax = sns.heatmap(wafermap.T, cmap=cmap, cbar=False, xticklabels=False, yticklabels=False, **kwargs)
    ax.grid(visible=True)
    ax.invert_yaxis()
    if title:
        ax.set_title(title)

    return fig if create_fig else None


def _build_default_allowed_bounds_map() -> dict[str, tuple[float, float]]:
    from endlesswm.trees import background_params, corruption_params, pattern_params

    merged = flatten_dict(background_params | pattern_params | corruption_params)
    return {
        key[:-2]: (float(val[0]), float(val[1]))
        for key, val in merged.items()
        if key.endswith("_r") and isinstance(val, (tuple, list)) and len(val) == 2
    }


def resolve_allowed_parameter_bounds(parameter_name: str) -> tuple[float, float] | None:
    bounds_map = _build_default_allowed_bounds_map()
    parts = parameter_name.split(".")
    for idx in range(len(parts)):
        candidate = ".".join(parts[idx:])
        if candidate in bounds_map:
            return bounds_map[candidate]
    return None


def resolve_scheduled_parameter_name(schedule_df: pd.DataFrame, parameter_name: str) -> str | None:
    parts = parameter_name.split(".")
    for idx in range(len(parts)):
        candidate = ".".join(parts[idx:])
        if f"{candidate}_r" in schedule_df.columns:
            return candidate
    return None


def resolve_numeric_parameter_names(
    parameter_df: pd.DataFrame,
    parameter_names: list[str] | None = None,
    *,
    include_constant_parameters: bool = False,
    excluded_name_fragments: tuple[str, ...] = (),
) -> tuple[list[str], list[str]]:
    selected_names = parameter_names or [column_name for column_name in parameter_df.columns if column_name not in {"task", "sample_idx"}]

    included_names: list[str] = []
    omitted_names: list[str] = []

    for name in selected_names:
        if excluded_name_fragments and any(fragment in name for fragment in excluded_name_fragments):
            omitted_names.append(name)
            continue

        if name not in parameter_df.columns:
            continue

        series = parameter_df[name]
        if not pd.api.types.is_numeric_dtype(series):
            omitted_names.append(name)
            continue

        valid_series = series.replace([np.inf, -np.inf], np.nan).dropna()
        if valid_series.empty:
            omitted_names.append(name)
            continue

        unique_values = set(valid_series.astype(float).unique().tolist())
        if unique_values.issubset({0.0, 1.0}):
            omitted_names.append(name)
            continue

        if include_constant_parameters:
            included_names.append(name)
            continue

        has_variation = int(valid_series.nunique(dropna=True)) > 1 and float(valid_series.max() - valid_series.min()) > 0.0
        if has_variation:
            included_names.append(name)
        else:
            omitted_names.append(name)

    return included_names, omitted_names


def build_parameter_evolution_matrix(
    parameter_df: pd.DataFrame,
    parameter_names: list[str] | None = None,
    statistic: Literal["median", "mean", "min", "max", "iqr_width"] = "median",
    include_constant_parameters: bool = False,
    excluded_name_fragments: tuple[str, ...] = (),
) -> pd.DataFrame:
    if "task" not in parameter_df.columns:
        raise ValueError("parameter_df must include a 'task' column for evolution summaries")

    names, _ = resolve_numeric_parameter_names(
        parameter_df,
        parameter_names,
        include_constant_parameters=include_constant_parameters,
        excluded_name_fragments=excluded_name_fragments,
    )
    if not names:
        return pd.DataFrame(index=sorted(parameter_df["task"].unique()), columns=[])

    grouped = parameter_df.groupby("task", sort=True)[names]
    if statistic == "median":
        matrix = grouped.median()
    elif statistic == "mean":
        matrix = grouped.mean()
    elif statistic == "min":
        matrix = grouped.min()
    elif statistic == "max":
        matrix = grouped.max()
    else:
        matrix = grouped.quantile(0.75) - grouped.quantile(0.25)

    return matrix.sort_index()


def build_parameter_evolution_heatmap(
    evolution_matrix: pd.DataFrame,
    title: str,
    cmap: str = "viridis",
    normalize_rows: bool = True,
    normalize_with_full_allowed_range: bool = False,
    show_parameter_labels: bool = False,
) -> Figure:
    fig, ax = plt.subplots(figsize=(18, 6))

    if evolution_matrix.empty:
        ax.text(0.5, 0.5, "No numeric parameters available", ha="center", va="center")
        ax.set_axis_off()
        return fig

    heatmap_matrix = evolution_matrix.T.copy()
    if normalize_rows:
        if normalize_with_full_allowed_range:
            bounds_map = _build_default_allowed_bounds_map()
            normalized_rows: dict[str, pd.Series] = {}
            for parameter_name, row in heatmap_matrix.iterrows():
                bounds = bounds_map.get(parameter_name)
                if bounds is not None and float(bounds[1]) > float(bounds[0]):
                    low_value, high_value = float(bounds[0]), float(bounds[1])
                    span_value = high_value - low_value
                    normalized_rows[parameter_name] = ((row - low_value) / span_value).clip(lower=0.0, upper=1.0)
                else:
                    row_min = float(row.min())
                    row_max = float(row.max())
                    row_span = row_max - row_min
                    if row_span <= 0.0:
                        normalized_rows[parameter_name] = row * 0.0
                    else:
                        normalized_rows[parameter_name] = (row - row_min) / row_span
            heatmap_matrix = pd.DataFrame.from_dict(normalized_rows, orient="index")

    sns.heatmap(heatmap_matrix, cmap=cmap, ax=ax, cbar=True)
    ax.set_xlabel("Task")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.tick_params(axis="y", left=False, labelleft=show_parameter_labels)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def build_single_parameter_schedule_figure(
    bounds_df: pd.DataFrame,
    parameter_name: str,
    *,
    title: str | None = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 4))

    if bounds_df.empty:
        ax.text(0.5, 0.5, "No schedule data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    tasks = bounds_df["task"].to_numpy()

    allowed_min = None
    allowed_max = None
    if {"allowed_min", "allowed_max"}.issubset(bounds_df.columns):
        allowed_min = bounds_df["allowed_min"].to_numpy(dtype=float)
        allowed_max = bounds_df["allowed_max"].to_numpy(dtype=float)
    else:
        allowed_bounds = resolve_allowed_parameter_bounds(parameter_name)
        if allowed_bounds is not None:
            allowed_min = np.full(len(tasks), float(allowed_bounds[0]), dtype=float)
            allowed_max = np.full(len(tasks), float(allowed_bounds[1]), dtype=float)

    if allowed_min is not None and allowed_max is not None:
        ax.fill_between(tasks, allowed_min, allowed_max, alpha=0.08, color="#7f7f7f", label="full range")
        ax.plot(tasks, allowed_min, color="#7f7f7f", linewidth=1.0, linestyle=":")
        ax.plot(tasks, allowed_max, color="#7f7f7f", linewidth=1.0, linestyle=":")

    if {"scheduled_min", "scheduled_max"}.issubset(bounds_df.columns):
        scheduled_min = bounds_df["scheduled_min"].to_numpy(dtype=float)
        scheduled_max = bounds_df["scheduled_max"].to_numpy(dtype=float)
        ax.fill_between(tasks, scheduled_min, scheduled_max, alpha=0.18, color="#4c78a8", label="scheduled range")
        ax.plot(tasks, scheduled_min, color="#4c78a8", linewidth=1.2, linestyle="--")
        ax.plot(tasks, scheduled_max, color="#4c78a8", linewidth=1.2, linestyle="--")

    if {"observed_min", "observed_max"}.issubset(bounds_df.columns):
        observed_min = bounds_df["observed_min"].to_numpy(dtype=float)
        observed_max = bounds_df["observed_max"].to_numpy(dtype=float)
        ax.fill_between(tasks, observed_min, observed_max, alpha=0.08, color="#f58518", label="observed range", zorder=3)
        ax.fill_between(
            tasks,
            observed_min,
            observed_max,
            facecolor="none",
            edgecolor="#f58518",
            hatch="||",
            linewidth=0.0,
            zorder=4,
        )
        ax.plot(tasks, observed_min, color="#f58518", linewidth=1.0, linestyle="-", marker="v", markersize=4, zorder=5)
        ax.plot(tasks, observed_max, color="#f58518", linewidth=1.0, linestyle="-", marker="^", markersize=4, zorder=5)
    if "observed_median" in bounds_df.columns:
        ax.plot(
            tasks,
            bounds_df["observed_median"].to_numpy(dtype=float),
            color="#e45756",
            linewidth=1.8,
            marker="o",
            markersize=3,
            label="observed median",
            zorder=6,
        )

    ax.set_xlabel("Task")
    ax.set_ylabel(parameter_name)
    if title:
        ax.set_title(title)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    return fig


def surface(
    wafermap: np.ndarray,
    title: str | None = None,
    create_fig: bool = True,  # noqa: FBT001, FBT002
    cmap: str = "terrain",
    offset: float = 0.2,
    voffset: float = 2.0,
    **kwargs: dict[str, Any],
) -> Figure | None:
    """🖼️ creates 3d surface visualization of a wafer map with contour projections on each plane.

    Args:
        wafermap (np.ndarray): 2d wafer map.
        title (str | None, optional): optional title for the heatmap. Defaults to None.
        create_fig (bool, optional): whether this function should create a new figure object. Defaults to True.
        cmap (str, optional): matplotlib colormap string. Defaults to "terrain".
        offset (float, optional): offset as a fraction of the width/height/value-range to add between the heatmap and
            the projections/top. Defaults to 0.2
        voffset (float, optional): offset as a fraction/multiple of the range of the values to put between
            the 3d heatmap and the projection on the x/y plane. Defaults to 2.0
        kwargs (dict[str, Any]): additional keyword arguments are passed to `seaborn.heatmap`.

    Returns:
        Figure | None: `create_fig` ? figure : None
    """
    _x = np.arange(wafermap.shape[0])
    _y = np.arange(wafermap.shape[1])
    _xx, _yy = np.meshgrid(_x, _y)

    xrange = wafermap.shape[0]
    dx = math.ceil(xrange * offset)
    xlim = (0 - dx, xrange + dx)
    yrange = wafermap.shape[1]
    dy = math.ceil(yrange * offset)
    ylim = (0 - dy, yrange + dy)
    vmin = np.nanmin(wafermap)
    vmax = np.nanmax(wafermap)
    vrange = vmax - vmin
    dv = math.ceil(vrange * offset)
    dvb = math.ceil(vrange * voffset)
    vlim = (vmin - dvb, vmax + dv)

    if create_fig:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    else:
        ax = kwargs.pop("ax")

    ax.set_proj_type("ortho")

    ax.plot_surface(_xx, _yy, wafermap.T, cmap=cmap, linewidth=0, antialiased=True, rstride=1, cstride=1)

    # Plot projections of the contours for each dimension.  By choosing offsets
    # that match the appropriate axes limits, the projected contours will sit on
    # the 'walls' of the graph
    ax.contourf(_xx, _yy, wafermap.T, zdir="z", offset=vlim[0], antialiased=True, cmap=cmap)
    ax.contourf(_xx, _yy, wafermap.T, zdir="x", offset=xlim[0], antialiased=True, cmap=f"{cmap}_r")
    ax.contourf(_xx, _yy, wafermap.T, zdir="y", offset=ylim[1], antialiased=True, cmap=f"{cmap}_r")

    ax.set(xlim=xlim, ylim=ylim, zlim=vlim)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    if title is not None:
        ax.set_title(title)

    return fig if create_fig else None


def plot_bar_3d(
    wafermap: np.ndarray,
    title: str | None = None,
    create_fig: bool = True,  # noqa: FBT001, FBT002
    cmap: str = "terrain",
    **kwargs: dict[str, Any],
) -> Figure | None:
    """🖼️ creates 3d bar chart/heatmap visualization of a wafer map.

    Args:
        wafermap (np.ndarray): 2d wafer map.
        title (str | None, optional): optional title for the heatmap. Defaults to None.
        create_fig (bool, optional): whether this function should create a new figure object. Defaults to True.
        cmap (str, optional): matplotlib colormap string. Defaults to "terrain".
        kwargs (dict[str, Any]): additional keyword arguments are passed to `seaborn.heatmap`.

    Returns:
        Figure | None: `create_fig` ? figure : None
    """
    wafermap = np.nan_to_num(wafermap, nan=0.0)

    if create_fig:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    else:
        ax = kwargs.pop("ax")

    jet = cm.get_cmap(cmap)
    colors = jet(wafermap.T.ravel())
    _x = np.arange(wafermap.shape[0])
    _y = np.arange(wafermap.shape[1])
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    bottom = np.zeros_like(wafermap.T.ravel())
    width = depth = 1
    ax.bar3d(x, y, bottom, width, depth, wafermap.T.ravel(), color=colors, shade=False, linewidths=0.1, alpha=0.7)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    if title is not None:
        ax.set_title(title)

    return fig if create_fig else None


def plot_histogram(wafermap: np.ndarray, title: str | None = None, create_fig: bool = True, **kwargs: dict[str, Any]) -> Figure | None:  # noqa: FBT001, FBT002
    """📊 creates a histogram visualization of a wafer map.

    Args:
        wafermap (np.ndarray): 2d wafer map.
        title (str | None, optional): optional title for the histogram. Defaults to None.
        create_fig (bool, optional): whether this function should create a new figure object. Defaults to True.
        kwargs (dict[str, Any]): additional keyword arguments are passed to `seaborn.histplot`.

    Returns:
        Figure | None: `create_fig` ? figure : None
    """
    if create_fig:
        fig = plt.figure()

    ax = sns.histplot(x=wafermap.flatten(), **kwargs)
    ax.grid(visible=True)

    bars = []
    bars.extend(ax.patches)  # Add the bars from this axis

    # Create a colormap
    cmap = plt.cm.jet

    # Get min and max values based on the x-positions (bar centers)
    # Extract x-centers of the bars
    x_centers = [bar.get_x() + bar.get_width() / 2 for bar in bars]
    min_val = x_centers[0]  # First bar's x-position
    max_val = x_centers[-1]  # Last bar's x-position

    # Normalize the values to the colormap range
    norm = plt.Normalize(vmin=min_val, vmax=max_val)

    # Iterate over each bar and change its color based on its x-position
    for bar in bars:
        x_pos = bar.get_x() + bar.get_width() / 2  # Get the x-center of the bar
        color = cmap(norm(x_pos))  # Map the x-position to the colormap
        bar.set_facecolor(color)  # Set the color of the bar

    if title is not None:
        ax.set_title(title)

    return fig if create_fig else None


def compact_histogram(wafermap: np.ndarray, title: str | None = None, create_fig: bool = True, **kwargs: dict[str, Any]) -> Figure | None:  # noqa: FBT001, FBT002
    """📊 creates a compact histogram visualization of a wafer map without y-ticks.

    Args:
        wafermap (np.ndarray): 2d wafer map.
        title (str | None, optional): optional title for the histogram. Defaults to None.
        create_fig (bool, optional): whether this function should create a new figure object. Defaults to True.
        kwargs (dict[str, Any]): additional keyword arguments are passed to `seaborn.histplot`.

    Returns:
        Figure | None: `create_fig` ? figure : None
    """
    if create_fig:
        fig = plt.figure()

    ax = sns.histplot(x=wafermap.flatten(), **kwargs)
    ax.grid(visible=True)

    bars = []
    bars.extend(ax.patches)  # Add the bars from this axis

    # Create a colormap
    cmap = plt.cm.jet

    # Get min and max values based on the x-positions (bar centers)
    # Extract x-centers of the bars
    x_centers = [bar.get_x() + bar.get_width() / 2 for bar in bars]
    min_val = x_centers[0]  # First bar's x-position
    max_val = x_centers[-1]  # Last bar's x-position

    # Normalize the values to the colormap range
    norm = plt.Normalize(vmin=min_val, vmax=max_val)

    # Iterate over each bar and change its color based on its x-position
    for bar in bars:
        x_pos = bar.get_x() + bar.get_width() / 2  # Get the x-center of the bar
        color = cmap(norm(x_pos))  # Map the x-position to the colormap
        bar.set_facecolor(color)  # Set the color of the bar

    if title is not None:
        ax.set_title(title)

    ax.set_yticks([])
    ax.set_ylabel("")

    return fig if create_fig else None


def plot_ecdf(wafermap: np.ndarray, title: str | None = None, create_fig: bool = True, **kwargs: dict[str, Any]) -> Figure | None:  # noqa: FBT001, FBT002
    """🖼️ creates an empirical cumulative distribution function (ECDF) visualization of a wafer map.

    Args:
        wafermap (np.ndarray): 2d wafer map.
        title (str | None, optional): optional title for the plot. Defaults to None.
        create_fig (bool, optional): whether this function should create a new figure object. Defaults to True.
        kwargs (dict[str, Any]): additional keyword arguments (`ax`, `linewidth`, `alpha`).

    Returns:
        Figure | None: `create_fig` ? figure : None
    """
    values = np.asarray(wafermap, dtype=float).flatten()
    values = values[np.isfinite(values)]

    if create_fig:
        fig, ax = plt.subplots()
    else:
        ax = kwargs.pop("ax")

    if values.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return fig if create_fig else None

    sorted_values = np.sort(values)
    proportions = np.arange(1, sorted_values.size + 1, dtype=float) / float(sorted_values.size)
    linewidth = float(kwargs.pop("linewidth", 2.0))
    alpha = float(kwargs.pop("alpha", 1.0))

    min_value = float(sorted_values[0])
    max_value = float(sorted_values[-1])
    if max_value <= min_value:
        max_value = min_value + 1.0
    norm = plt.Normalize(vmin=min_value, vmax=max_value)
    cmap = plt.cm.jet

    if sorted_values.size > 1:
        segment_step = max(1, sorted_values.size // 512)
        for start_index in range(0, sorted_values.size - 1, segment_step):
            end_index = min(start_index + segment_step, sorted_values.size - 1)
            segment_x = proportions[start_index : end_index + 1]
            segment_y = sorted_values[start_index : end_index + 1]
            color_index = min((start_index + end_index) // 2, sorted_values.size - 1)
            ax.plot(
                segment_x,
                segment_y,
                color=cmap(norm(sorted_values[color_index])),
                linewidth=linewidth,
                alpha=alpha,
            )

    marker_step = max(1, sorted_values.size // 128)
    ax.scatter(
        proportions[::marker_step],
        sorted_values[::marker_step],
        c=sorted_values[::marker_step],
        cmap=cmap,
        norm=norm,
        s=10,
        linewidths=0,
        alpha=min(alpha + 0.1, 1.0),
    )

    y_padding = max((max_value - min_value) * 0.03, 1e-6)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(min_value - y_padding, max_value + y_padding)
    ax.set_xlabel("")
    ax.grid(visible=True)

    if title is not None:
        ax.set_title(title)

    return fig if create_fig else None


def plot_sinogram(wafermap: np.ndarray, title: str | None = "Sinogram", create_fig: bool = True, **kwargs: dict[str, Any]) -> Figure | None:  # noqa: FBT001, FBT002
    """🖼️ creates a Radon sinogram visualization of a wafer map.

    Args:
        wafermap (np.ndarray): 2d wafer map.
        title (str | None, optional): optional title for the sinogram. Defaults to "Sinogram".
        create_fig (bool, optional): whether this function should create a new figure object. Defaults to True.
        kwargs (dict[str, Any]): additional keyword arguments are passed to `seaborn.heatmap`.

    Returns:
        Figure | None: `create_fig` ? figure : None
    """
    # replace NaN with 0.0
    wafermap = np.nan_to_num(wafermap, nan=0.0)

    sinogram = radon(wafermap, preserve_range=True)

    if create_fig:
        fig = plt.figure()

    ax = sns.heatmap(sinogram, cmap="mako", cbar=False, **kwargs)
    ax.grid(visible=True)
    ax.invert_yaxis()
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("projection angle [deg]")
    ax.set_ylabel("projection position [pixel]")

    return fig if create_fig else None


def plot_compact_sinogram(
    wafermap: np.ndarray,
    title: str | None = "Sinogram",
    create_fig: bool = True,  # noqa: FBT001, FBT002
    **kwargs: dict[str, Any],
) -> Figure | None:
    """🖼️ creates a compact Radon sinogram visualization of a wafer map without ticks and cbar.

    Args:
        wafermap (np.ndarray): 2d wafer map.
        title (str | None, optional): optional title for the sinogram. Defaults to "Sinogram".
        create_fig (bool, optional): whether this function should create a new figure object. Defaults to True.
        kwargs (dict[str, Any]): additional keyword arguments are passed to `seaborn.heatmap`.

    Returns:
        Figure | None: `create_fig` ? figure : None
    """
    # replace NaN with 0.0
    wafermap = np.nan_to_num(wafermap, nan=0.0)

    sinogram = radon(wafermap, preserve_range=True)

    if create_fig:
        fig = plt.figure()

    ax = sns.heatmap(sinogram, cmap="mako", cbar=False, xticklabels=False, yticklabels=False, **kwargs)
    ax.grid(visible=True)
    ax.invert_yaxis()
    if title is not None:
        ax.set_title(title)

    return fig if create_fig else None


def plot_fft(
    wafermap: np.ndarray,
    title: str | None = "Fourier Transform",
    create_fig: bool = True,  # noqa: FBT001, FBT002
    **kwargs: dict[str, Any],
) -> Figure | None:
    """🖼️ creates a 2d fft visualization of a wafer map.

    Args:
        wafermap (np.ndarray): 2d wafer map.
        title (str | None, optional): optional title for the heatmap. Defaults to "Fourier Transform".
        create_fig (bool, optional): whether this function should create a new figure object. Defaults to True.
        kwargs (dict[str, Any]): additional keyword arguments are passed to `seaborn.heatmap`.

    Returns:
        Figure | None: `create_fig` ? figure : None
    """
    # replace NaN with 0.0
    wafermap = np.nan_to_num(wafermap, nan=0.0)

    fft = np.log10(np.abs(fftshift(fft2(wafermap))) ** 2)

    if create_fig:
        fig = plt.figure()

    ax = sns.heatmap(fft, cmap="magma", cbar=False, **kwargs)
    ax.grid(visible=True)
    ax.invert_yaxis()
    if title is not None:
        ax.set_title(title)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    return fig if create_fig else None


def plot_xfft(
    wafermap: np.ndarray,
    title: str | None = "Fourier Transform in x direction",
    create_fig: bool = True,  # noqa: FBT001, FBT002
    **kwargs: dict[str, Any],
) -> Figure | None:
    """🖼️ creates a sliced 2d fft visualization of a wafer map in x direction.

    Args:
        wafermap (np.ndarray): 2d wafer map.
        title (str | None, optional): optional title for the plot. Defaults to "Fourier Transform in x direction".
        create_fig (bool, optional): whether this function should create a new figure object. Defaults to True.
        kwargs (dict[str, Any]): additional keyword arguments are passed to `seaborn.lineplot`.

    Returns:
        Figure | None: `create_fig` ? figure : None
    """
    # replace NaN with 0.0
    wafermap = np.nan_to_num(wafermap, nan=0.0)

    fft = np.log10(np.abs(fftshift(fft2(wafermap))) ** 2)
    xfft = fft[fft.shape[0] // 2 :, fft.shape[1] // 2]

    if create_fig:
        fig = plt.figure()

    ax = sns.lineplot(xfft, **kwargs)
    ax.grid(visible=True)
    if title is not None:
        ax.set_title(title)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    return fig if create_fig else None


def wafer_map_analysis(wafermap: np.ndarray) -> Figure:
    """🖼️ creates a figure with many wafer map visualizations."""
    fig, axs = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")

    compact_heatmap(wafermap, ax=axs[0], title="Wafer Map")
    plot_fft(wafermap, ax=axs[1])
    plot_compact_sinogram(wafermap, ax=axs[2])

    return fig


def animate_wafer_map_analysis(
    samples: list,
    interval: int = 200,
    filepath: str | None = None,
) -> IPython.display.HTML:
    """🎞️ plots all `samples` and creates an animation.

    Args:
        samples (list): list of samples
        interval (int, optional): delay between frames in milliseconds. Defaults to 200.
        filepath (str | None, optional): if a filepath is given, save the animation as a gif. Defaults to None.

    Returns:
        IPython.display.HTML: animation player as an ipython HTML output
    """
    with plt.ioff():
        fig, axs = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")

        def _plotting_func(idx: int, axs: plt.Axes) -> None:
            for ax in axs:
                ax.clear()
            compact_heatmap(samples[idx], ax=axs[0], title="Wafer Map")
            plot_fft(samples[idx], ax=axs[1])
            plot_compact_sinogram(samples[idx], ax=axs[2])

        anim = animation.FuncAnimation(fig, _plotting_func, frames=len(samples), fargs=(axs,), interval=interval)

        if filepath:
            writer = animation.PillowWriter(fps=math.ceil(1.0 / (interval * 0.001)), bitrate=1800)
            anim.save(f"{filepath}.gif", writer=writer)

        return IPython.display.HTML(anim.to_jshtml())


def wafer_map_analysis_small(wafermap: np.ndarray) -> Figure:
    """🖼️ creates a figure with many wafer map visualizations."""
    fig, axs = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")

    compact_heatmap(wafermap, ax=axs[0], title="Wafer Map")
    plot_ecdf(wafermap, ax=axs[1], title="ECDF")
    plot_histogram(wafermap, ax=axs[2], title="Histogram")

    return fig


def wafer_map_analysis_dist(wafermap: np.ndarray) -> Figure:
    """🖼️ creates a figure with many wafer map visualizations."""
    fig, axs = plt.subplots(1, 5, figsize=(16, 3), layout="constrained")

    compact_heatmap(wafermap, ax=axs[0], title="Wafer Map")
    plot_fft(wafermap, ax=axs[1])
    plot_compact_sinogram(wafermap, ax=axs[2])
    plot_ecdf(wafermap, ax=axs[3], title="ECDF")
    plot_histogram(wafermap, ax=axs[4], title="Histogram")

    return fig


def animate_wafer_map_analysis_dist(
    samples: list,
    interval: int = 200,
    filepath: str | None = None,
) -> IPython.display.HTML:
    """🎞️ plots all `samples` and creates an animation.

    Args:
        samples (list): list of samples
        interval (int, optional): delay between frames in milliseconds. Defaults to 200.
        filepath (str | None, optional): if a filepath is given, save the animation as a gif. Defaults to None.

    Returns:
        IPython.display.HTML: animation player as an ipython HTML output
    """
    with plt.ioff():
        fig, axs = plt.subplots(1, 5, figsize=(16, 3), layout="constrained")

        def _plotting_func(idx: int, axs: plt.Axes) -> None:
            for ax in axs:
                ax.clear()
            compact_heatmap(samples[idx], ax=axs[0], title="Wafer Map")
            plot_fft(samples[idx], ax=axs[1])
            plot_compact_sinogram(samples[idx], ax=axs[2])
            plot_ecdf(samples[idx], ax=axs[3], title="ECDF")
            plot_histogram(samples[idx], ax=axs[4], title="Histogram")

        anim = animation.FuncAnimation(fig, _plotting_func, frames=len(samples), fargs=(axs,), interval=interval)

        if filepath:
            writer = animation.PillowWriter(fps=math.ceil(1.0 / (interval * 0.001)), bitrate=1800)
            anim.save(f"{filepath}.gif", writer=writer)

        return IPython.display.HTML(anim.to_jshtml())


def wafer_map_analysis_3d(wafermap: np.ndarray) -> Figure:
    """🖼️ creates a figure with many wafer map visualizations."""
    # Create a figure and a gridspec object
    fig, ax = plt.subplot_mosaic(
        [["a", "b", "e", "e"], ["c", "d", "e", "e"]],
        figsize=(12, 6),
        layout="constrained",
        per_subplot_kw={"e": {"projection": "3d"}},
    )

    compact_heatmap(wafermap, ax=ax["a"], title="Wafer Map")
    plot_fft(wafermap, ax=ax["d"])
    plot_compact_sinogram(wafermap, ax=ax["c"])
    plot_histogram(wafermap, ax=ax["b"], title="Histogram")
    surface(wafermap, ax=ax["e"], create_fig=False)

    return fig


def animate_wafer_map_analysis_3d(
    samples: list,
    interval: int = 200,
    filepath: str | None = None,
) -> IPython.display.HTML:
    """🎞️ plots all `samples` and creates an animation.

    Args:
        samples (list): list of samples
        interval (int, optional): delay between frames in milliseconds. Defaults to 200.
        filepath (str | None, optional): if a filepath is given, save the animation as a gif. Defaults to None.

    Returns:
        IPython.display.HTML: animation player as an ipython HTML output
    """
    with plt.ioff():
        fig, ax = plt.subplot_mosaic(
            [["a", "b", "e", "e"], ["c", "d", "e", "e"]],
            figsize=(12, 6),
            layout="constrained",
            per_subplot_kw={"e": {"projection": "3d"}},
        )

        def _plotting_func(idx: int, ax: plt.Axes) -> None:
            for ax_ in ax.values():
                ax_.clear()
            compact_heatmap(samples[idx], ax=ax["a"], title="Wafer Map")
            plot_fft(samples[idx], ax=ax["d"])
            plot_compact_sinogram(samples[idx], ax=ax["c"])
            plot_histogram(samples[idx], ax=ax["b"], title="Histogram")
            surface(samples[idx], ax=ax["e"], create_fig=False)

        anim = animation.FuncAnimation(fig, _plotting_func, frames=len(samples), fargs=(ax,), interval=interval)

        if filepath:
            writer = animation.PillowWriter(fps=math.ceil(1.0 / (interval * 0.001)), bitrate=1800)
            anim.save(f"{filepath}.gif", writer=writer)

        return IPython.display.HTML(anim.to_jshtml())


def plot_grid(
    samples: list[Any],
    plotting_func: Callable[[Any, plt.Axes], plt.Axes],
    columns: int | None = None,
    titles: list[str] | None = None,
) -> mpl.figure.Figure:
    """📟 plots a grid of samples.

    Args:
        samples (list[Any]): samples to pass to `plotting_func` as first argument each.
        plotting_func (Callable[[Any, plt.Axes], plt.Axes]): function to visualize a single sample.
            The function should take a sample and a `plt.Axes` as arguments.
        columns (int, optional): number of samples in the columns (and rows).
            If None, use the ceiled square root of the number of samples. Defaults to None.
        titles (list[str] | None, optional): titles for each sample. Defaults to None.

    Returns:
        mpl.figure.Figure: matplotlib figure object
    """
    if columns is None:
        columns = math.ceil(math.sqrt(len(samples)))

    with plt.ioff():
        fig = plt.figure()

        for idx, sample in enumerate(samples):
            ax = fig.add_subplot(columns, columns, idx + 1)
            _ = plotting_func(sample, ax=ax)
            if titles:
                ax.set_title(titles[idx])

        plt.tight_layout()

        if titles:
            fig.subplots_adjust(hspace=0.15, wspace=0.05)
        else:
            fig.subplots_adjust(hspace=0.05, wspace=0.05)

        return fig


def animate(
    samples: list,
    plotting_func: Callable[[Any, plt.Axes], plt.Axes],
    interval: int = 200,
    filepath: str | None = None,
) -> IPython.display.HTML:
    """🎞️ plots all `samples` and creates an animation.

    Args:
        samples (list): list of samples
        plotting_func (Callable[[Any, plt.Axes], plt.Axes]): function to visualize a single sample.
            The function should take a sample and a `plt.Axes` as arguments.
        interval (int, optional): delay between frames in milliseconds. Defaults to 200.
        filepath (str | None, optional): if a filepath is given, save the animation as a gif. Defaults to None.

    Returns:
        IPython.display.HTML: animation player as an ipython HTML output
    """
    with plt.ioff():
        fig, ax = plt.subplots()

        def _plotting_func(idx: int, ax: plt.Axes) -> None:
            ax.clear()
            plotting_func(samples[idx], ax=ax)

        anim = animation.FuncAnimation(fig, _plotting_func, frames=len(samples), fargs=(ax,), interval=interval)

        if filepath:
            writer = animation.PillowWriter(fps=math.ceil(1.0 / (interval * 0.001)), bitrate=1800)
            anim.save(f"{filepath}.gif", writer=writer)

        return IPython.display.HTML(anim.to_jshtml())


def animate_grid(
    samples: list,
    plotting_func: Callable[[Any, plt.Axes], plt.Axes],
    samples_per_frame: int = 16,
    columns: int | None = None,
    rows: int | None = None,
    interval: int = 200,
    filepath: str | None = None,
) -> IPython.display.HTML:
    """🎞️ plots all `samples` in a grid and creates an animation.

    Args:
        samples (list): list of samples
        plotting_func (Callable[[Any, plt.Axes], plt.Axes]): function to visualize a single sample.
            The function should take a sample and a `plt.Axes` as arguments.
        samples_per_frame (int, optional): number of samples to show in each grid/frame. Defaults to 16.
        columns (int | None, optional): number of grid columns per frame.
        rows (int | None, optional): number of grid rows per frame.
        interval (int, optional): delay between frames in milliseconds. Defaults to 200.
        filepath (str | None, optional): if a filepath is given, save the animation as a gif. Defaults to None.

    Returns:
        IPython.display.HTML: animation player as an ipython HTML output
    """
    if columns is None:
        columns = math.ceil(math.sqrt(samples_per_frame))
    if rows is None:
        rows = columns

    with plt.ioff():
        fig = plt.figure(figsize=(columns * 2, rows * 2))
        axs = [fig.add_subplot(rows, columns, idx + 1) for idx in range(samples_per_frame)]

        def _plotting_func(idx: int, axs: list[plt.Axes]) -> None:
            for i in range(samples_per_frame):
                axs[i].clear()
                if idx * samples_per_frame + i < len(samples):
                    plotting_func(samples[idx * samples_per_frame + i], ax=axs[i])

        plt.tight_layout()
        fig.subplots_adjust(hspace=0.05, wspace=0.05)

        anim = animation.FuncAnimation(
            fig,
            _plotting_func,
            frames=math.ceil(len(samples) / samples_per_frame),
            fargs=(axs,),
            interval=interval,
        )

        if filepath:
            writer = animation.PillowWriter(fps=math.ceil(1.0 / (interval * 0.001)), bitrate=1800)
            anim.save(f"{filepath}.gif", writer=writer)

        return IPython.display.HTML(anim.to_jshtml())


def plot_multilabel_distribution(
    dset: H5WM,
) -> Figure:
    """📊 plots a multilabel distribution using matplotlib and seaborn.

    Args:
        dset (H5WM): dataset containing multi-label targets.

    Returns:
        Figure: matplotlib figure instance.
    """
    targets = dset.labels
    patterns = classes
    patterns_str = [str(pat) for pat in patterns]

    num_samples, num_classes = targets.shape
    no_pattern_name = r"$\emptyset$"

    # Per-sample cardinality
    cardinality = targets.sum(dim=1).long()
    max_cardinality = int(cardinality.max().item())

    data = []
    for sample_idx in range(num_samples):
        k = int(cardinality[sample_idx].item())
        if k == 0:
            # Represent "all-negative" samples as a special pattern
            data.append(
                {
                    "sample": sample_idx,
                    "pattern": no_pattern_name,
                    "index": -1,
                    "cardinality": 0,
                },
            )
            continue
        for class_idx in range(num_classes):
            if targets[sample_idx, class_idx] == 1.0:
                data.append(
                    {
                        "sample": sample_idx,
                        "pattern": patterns_str[class_idx],
                        "index": class_idx,
                        "cardinality": k,
                    },
                )
    labels_df = pl.DataFrame(data)

    cardinality_matrix = np.zeros((num_classes + 1, max_cardinality + 1), dtype=float)
    for sample_idx in range(num_samples):
        k = int(cardinality[sample_idx])
        if k == 0:
            cardinality_matrix[0, 0] += 1
        for class_idx in range(num_classes):
            if targets[sample_idx, class_idx] == 1.0:
                cardinality_matrix[class_idx + 1, k] += 1

    co_occurrence_matrix = np.zeros((num_classes, num_classes), dtype=float)
    for sample_idx in range(num_samples):
        if cardinality[sample_idx] == 0:
            continue
        for class_idx in range(num_classes):
            if targets[sample_idx, class_idx] == 1.0:
                for co_occurrence_idx in range(num_classes):
                    if targets[sample_idx, co_occurrence_idx] == 1.0:
                        co_occurrence_matrix[class_idx, co_occurrence_idx] += 1

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle("Multi-Label Distribution")

    # Pattern-Cardinality Heatmap
    axes[0, 0].set_title("Class-Cardinality Matrix")
    sns.heatmap(
        cardinality_matrix,
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        yticklabels=[no_pattern_name, *patterns_str],
        cbar=None,
        cmap=sns.color_palette("light:b", as_cmap=True),
        ax=axes[0, 0],
    )

    # Pattern Histogram (per-label counts)
    sns.countplot(
        labels_df.to_pandas(),
        y="pattern",
        hue="cardinality",
        order=[no_pattern_name, *patterns_str],
        dodge=True,
        ax=axes[0, 1],
        palette=sns.color_palette("Spectral", n_colors=max(max_cardinality + 1, 1)),
    )
    sns.despine(left=True, bottom=False, ax=axes[0, 1])

    # Cardinality Histogram (per-sample counts, grouped by class) — debiased via weights
    cdf = labels_df.to_pandas()
    # Each sample with k>0 contributes k rows; weight each row by 1/k so totals equal the number of samples
    cdf["weight"] = 1.0
    mask = cdf["cardinality"] > 0
    cdf.loc[mask, "weight"] = 1.0 / cdf.loc[mask, "cardinality"]

    sns.histplot(
        cdf,
        x="cardinality",
        hue="pattern",
        hue_order=[no_pattern_name, *patterns_str],
        weights="weight",
        multiple="stack",  # or "stack" if you prefer stacked bars
        discrete=True,  # treat cardinality as discrete bins
        binrange=(0, max_cardinality),
        ax=axes[1, 0],
    )
    axes[1, 0].set_xlabel("cardinality")
    axes[1, 0].set_ylabel("count")
    sns.despine(left=False, bottom=True, ax=axes[1, 0])

    # Co-Occurrence Matrix
    axes[1, 1].set_title("Co-Occurrence Matrix")
    sns.heatmap(
        co_occurrence_matrix,
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        xticklabels=patterns_str,
        yticklabels=patterns_str,
        cbar=None,
        cmap=sns.color_palette("light:b", as_cmap=True),
        ax=axes[1, 1],
    )

    plt.tight_layout()
    return fig


def plot_class_cardinality_matrix(
    targets: torch.Tensor,
    title: str | None = None,
    **kwargs,
) -> Figure:
    cardinality_matrix = compute_cardinality_count_matrix(targets)[1:, 1:]

    f = plt.figure()
    ax = f.add_subplot(111)

    pos = ax.imshow(cardinality_matrix, interpolation="nearest", cmap="Blues", **kwargs)
    f.colorbar(pos, ax=ax, shrink=0.7)
    plt.grid(False)

    plt.title(title)

    ax.set_xticks([])
    ax.set_yticks([])

    plt.ylabel("class index")
    plt.xlabel("cardinality")

    return f


def plot_co_occurrence_matrix(
    targets: torch.Tensor,
    title: str | None = None,
    **kwargs,
) -> Figure:
    cardinality_matrix = compute_co_occurrence_matrix(targets)

    f = plt.figure()
    ax = f.add_subplot(111)

    pos = ax.imshow(cardinality_matrix, interpolation="nearest", cmap="Blues", **kwargs)
    f.colorbar(pos, ax=ax, shrink=0.7)
    plt.grid(False)

    plt.title(title)

    ax.set_xticks([])
    ax.set_yticks([])

    plt.ylabel("class index")
    plt.xlabel("class index")

    return f


def plot_cardinality_histogram(
    targets: torch.Tensor,
    title: str | None = None,
    **kwargs,
) -> Figure:
    cardinality = targets.sum(dim=1)

    values = cardinality.detach().cpu().numpy()
    df = pd.DataFrame({"cardinality": values})

    # Ensure numeric ordering on the x-axis
    order = sorted(pd.unique(df["cardinality"]))

    # Plot
    f = plt.figure()
    ax = sns.countplot(df, x="cardinality", order=order, **kwargs)
    if title:
        ax.set_title(title)
    sns.despine(left=False, bottom=True, ax=ax)

    return f


def plot_class_distribution_over_tasks(
    dset: H5WM,
) -> Figure:
    labels = [d.labels for d in dset.tasks]

    dfs = []
    for idx, l in enumerate(labels):
        df = compute_long_form_targets_dataframe(l, classes)
        df["task"] = idx
        dfs.append(df)
    df = pd.concat(dfs)

    g = sns.displot(
        data=df,
        x="task",
        hue="pattern",
        kind="hist",
        multiple="fill",
        hue_order=[r"$\emptyset$", *classes],
        bins=len(labels),
        aspect=1.6,
    )

    g.set(ylabel="")

    return g.fig


def plot_cardinality_distribution_over_tasks(
    dset: H5WM,
) -> Figure:
    labels = [d.labels for d in dset.tasks]

    dfs = []
    for idx, l in enumerate(labels):
        cardinality_values = compute_cardinality(l).detach().cpu().numpy().astype(int)
        task_df = pd.DataFrame(
            {
                "task": np.full(cardinality_values.shape[0], idx, dtype=int),
                "cardinality": cardinality_values,
            },
        )
        dfs.append(task_df)
    df = pd.concat(dfs)

    hue_order = sorted(int(v) for v in pd.unique(df["cardinality"]))

    fig = sns.displot(
        data=df,
        x="task",
        hue="cardinality",
        kind="hist",
        multiple="fill",
        hue_order=hue_order,
        bins=len(labels),
        aspect=1.6,
        palette=sns.color_palette("Spectral", n_colors=max(len(hue_order), 1)),
    )

    fig.set(ylabel="")

    return fig.fig


def plot_snr_over_stream(
    hdf5_paths: str | dict[str, str],
    snr_kind: Literal["energy", "variance"] = "energy",
    db: bool = True,
    max_samples_per_task: int | None = None,
    quantiles: tuple[float, float] = (0.25, 0.75),
    figsize: tuple[float, float] = (10.0, 4.5),
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> Figure:
    """📡📈 Plot per-task SNR distribution as a function of task index.

    For each task in each HDF5 dataset, computes the per-sample SNR
    (`background` vs `pattern`) and plots the median (line) plus an
    inter-quantile band.

    Args:
        hdf5_paths: either a single path (one dataset) or a `dict
            label -> path` for overlaying multiple datasets on the same
            axes.
        snr_kind: `"energy"` for `E[p²]/E[b²]`, `"variance"` for
            `var(p)/var(b)` (insensitive to additive offsets).
        db: report SNR in decibels (`10*log10`). Default True.
        max_samples_per_task: cap samples per task for speed (None = all).
        quantiles: lower/upper quantile pair for the shaded band.
            Default `(0.25, 0.75)` (IQR).
        figsize: matplotlib figsize, used only when `ax is None`.
        title: figure title.
        ax: pre-existing axes to plot into. When given, `figsize` is
            ignored and the parent figure is returned.

    Returns:
        The matplotlib Figure.
    """
    from pathlib import Path as _Path

    import h5py

    from .io.snr import snr_energy, snr_variance

    if isinstance(hdf5_paths, str):
        paths = {_Path(hdf5_paths).stem: hdf5_paths}
    else:
        paths = dict(hdf5_paths)

    snr_fn = snr_energy if snr_kind == "energy" else snr_variance
    q_lo, q_hi = float(quantiles[0]), float(quantiles[1])

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for label, path in paths.items():
        task_idx: list[int] = []
        medians: list[float] = []
        lo_band: list[float] = []
        hi_band: list[float] = []
        with h5py.File(str(path), "r") as f:
            task_keys = sorted(int(k) for k in f["tasks"].keys())
            for tk in task_keys:
                t = f[f"tasks/{tk}"]
                n = int(t["sample"].shape[0])
                if max_samples_per_task is not None:
                    n = min(n, max_samples_per_task)
                bg = t["background"][:n]
                pat = t["pattern"][:n]
                vals = np.array([snr_fn(bg[i], pat[i]) for i in range(n)], dtype=float)
                if db:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        vals = 10.0 * np.log10(vals)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                task_idx.append(int(tk))
                medians.append(float(np.median(vals)))
                lo_band.append(float(np.quantile(vals, q_lo)))
                hi_band.append(float(np.quantile(vals, q_hi)))

        x = np.array(task_idx)
        ax.plot(x, medians, marker="o", markersize=3, linewidth=1.6, label=str(label))
        ax.fill_between(x, lo_band, hi_band, alpha=0.18)

    ylabel = "SNR [dB]" if db else "SNR"
    ax.set_xlabel("task")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if len(paths) > 1:
        ax.legend(loc="best", fontsize=9)
    if title:
        ax.set_title(title)
    elif len(paths) == 1:
        ax.set_title(f"SNR over stream — {next(iter(paths))}")
    fig.tight_layout()
    return fig
