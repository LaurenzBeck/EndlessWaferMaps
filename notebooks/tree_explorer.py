# ruff: noqa: ANN, B018, E402, INP001, I001

"""🌳 Marimo tree explorer."""

import marimo

__generated_with = "0.22.0"
app = marimo.App(width="full", app_title="🌳 Tree Explorer")


@app.cell
def _():
    import inspect
    import pathlib

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from endlesswm.explorer_support import build_pattern_corruption_kwargs, build_pattern_generator_kwargs, widget_values_to_kwargs
    from endlesswm.geometry import full_ellipse, linear_gradient, linear_radial_gradient
    from endlesswm.noise import beta_weibull, impulse_noise, pink_noise
    from endlesswm.patterns import (
        add_to_mask,
        add_to_pattern,
        classes,
        corruption_params,
        pattern_corruptions,
        pattern_generators,
        pattern_params,
    )
    from endlesswm.pipeline.config.loader import (
        list_available_components,
        load_component_config,
    )
    from endlesswm.pipeline.config.schema import ConceptDriftConfig, CovariateDriftConfig, DriftConfig, PriorDriftConfig
    from endlesswm.pipeline.tree_factory import TreeFactory
    from endlesswm.transforms import gaussian_filter
    from endlesswm.trees import background_params, combine_signals, init_output_dictionary, mask_sample
    from endlesswm.utils import affine
    from endlesswm.visualizations import compact_heatmap, plot_compact_sinogram, plot_ecdf, plot_fft, plot_histogram, plot_mask

    return (
        ConceptDriftConfig,
        CovariateDriftConfig,
        DriftConfig,
        PriorDriftConfig,
        TreeFactory,
        add_to_mask,
        add_to_pattern,
        affine,
        background_params,
        beta_weibull,
        build_pattern_corruption_kwargs,
        build_pattern_generator_kwargs,
        classes,
        combine_signals,
        compact_heatmap,
        corruption_params,
        full_ellipse,
        gaussian_filter,
        impulse_noise,
        init_output_dictionary,
        inspect,
        pathlib,
        linear_gradient,
        linear_radial_gradient,
        list_available_components,
        load_component_config,
        mask_sample,
        mo,
        np,
        pattern_corruptions,
        pattern_generators,
        pattern_params,
        pink_noise,
        plot_compact_sinogram,
        plot_ecdf,
        plot_fft,
        plot_histogram,
        plot_mask,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 🌳 EndlessWM - Tree Explorer
    """)
    return


@app.cell
def _(classes, mo):
    tree_height_widget = mo.ui.number(start=16, stop=256, step=1, value=52, label="height")
    tree_width_widget = mo.ui.number(start=16, stop=256, step=1, value=52, label="width")
    tree_single_class_widget = mo.ui.dropdown(
        {class_name: class_name for class_name in classes},
        value="center_spot",
        label="single-label class",
    )
    tree_seed_widget = mo.ui.number(start=0, stop=100000, step=1, value=42, label="seed")
    tree_save_plots_widget = mo.ui.checkbox(value=False, label="save plots")
    tree_save_plots_directory_widget = mo.ui.text(value="plots/tree_explorer", label="plot directory")
    _tree_top_section = mo.vstack(
        [
            mo.md("## 📐 Wafer Map Size"),
            mo.hstack([tree_height_widget, tree_width_widget], justify="start", gap=1),
            mo.md("## 🌐 step-by-step sample transformation"),
            mo.hstack([tree_single_class_widget, tree_seed_widget], justify="start", gap=1, wrap=True),
            mo.hstack([tree_save_plots_widget, tree_save_plots_directory_widget], justify="start", gap=1, wrap=True),
        ],
        gap=1,
    )
    _tree_top_section
    return (
        tree_height_widget,
        tree_save_plots_directory_widget,
        tree_save_plots_widget,
        tree_seed_widget,
        tree_single_class_widget,
        tree_width_widget,
        widget_values_to_kwargs,
    )


@app.cell
def _(pathlib, tree_save_plots_directory_widget, tree_save_plots_widget):
    tree_save_plots_enabled = bool(tree_save_plots_widget.value)
    _tree_directory_text = str(tree_save_plots_directory_widget.value).strip()
    tree_save_plots_directory = pathlib.Path(_tree_directory_text).expanduser() if _tree_directory_text else None
    if tree_save_plots_enabled and tree_save_plots_directory is not None:
        tree_save_plots_directory.mkdir(parents=True, exist_ok=True)
    return tree_save_plots_directory, tree_save_plots_enabled


@app.cell
def _(background_params, mo):
    tree_background_widget_map = {}
    tree_background_key_maps = {}
    for _tree_node_name, _tree_parameter_bounds in background_params.items():
        _tree_node_widget_map = {}
        _tree_node_key_map = {}
        for _tree_parameter_name, _tree_bounds in _tree_parameter_bounds.items():
            _tree_display_name = _tree_parameter_name[:-2] if _tree_parameter_name.endswith(("_r", "_c")) else _tree_parameter_name
            _tree_node_key_map[_tree_display_name] = _tree_parameter_name
            if _tree_parameter_name.endswith("_r"):
                _tree_low_value = float(_tree_bounds[0])
                _tree_high_value = float(_tree_bounds[1])
                _tree_node_widget_map[_tree_display_name] = mo.ui.slider(
                    start=_tree_low_value,
                    stop=_tree_high_value,
                    step=max((_tree_high_value - _tree_low_value) / 200.0, 0.0005),
                    value=float((_tree_low_value + _tree_high_value) / 2.0),
                    show_value=True,
                    include_input=True,
                )
        tree_background_widget_map[_tree_node_name] = mo.ui.dictionary(_tree_node_widget_map, label="parameters")
        tree_background_key_maps[_tree_node_name] = _tree_node_key_map
    tree_background_controls = mo.ui.dictionary(tree_background_widget_map)
    return tree_background_controls, tree_background_key_maps, tree_background_widget_map


@app.cell
def _(mo, pattern_params, tree_single_class_widget):
    _tree_pattern_widget_map = {}
    _tree_pattern_key_map = {}
    for _tree_parameter_name, _tree_bounds in pattern_params[tree_single_class_widget.value].items():
        _tree_display_name = _tree_parameter_name[:-2] if _tree_parameter_name.endswith(("_r", "_c")) else _tree_parameter_name
        _tree_pattern_key_map[_tree_display_name] = _tree_parameter_name
        if _tree_parameter_name.endswith("_r"):
            _tree_low_value = float(_tree_bounds[0])
            _tree_high_value = float(_tree_bounds[1])
            _tree_pattern_widget_map[_tree_display_name] = mo.ui.slider(
                start=_tree_low_value,
                stop=_tree_high_value,
                step=max((_tree_high_value - _tree_low_value) / 200.0, 0.0005),
                value=float((_tree_low_value + _tree_high_value) / 2.0),
                show_value=True,
                include_input=True,
            )
        elif _tree_parameter_name.endswith("_c"):
            _tree_initial_option_name = str(_tree_bounds[0]).lower() if isinstance(_tree_bounds[0], bool) else str(_tree_bounds[0])
            _tree_pattern_widget_map[_tree_display_name] = mo.ui.dropdown(
                {
                    str(_tree_choice).lower() if isinstance(_tree_choice, bool) else str(_tree_choice): _tree_choice
                    for _tree_choice in _tree_bounds
                },
                value=_tree_initial_option_name,
            )
    tree_pattern_parameter_key_map = _tree_pattern_key_map
    tree_pattern_parameter_widget = mo.ui.dictionary(_tree_pattern_widget_map, label="parameters")
    return tree_pattern_parameter_key_map, tree_pattern_parameter_widget


@app.cell
def _(corruption_params, mo, pattern_corruptions, tree_single_class_widget):
    _tree_corruption_widget_map = {}
    _tree_corruption_key_maps = {}
    for _tree_corruption in pattern_corruptions[tree_single_class_widget.value]:
        _tree_corruption_name = _tree_corruption.__name__
        _tree_parameter_widget_map = {}
        _tree_parameter_key_map = {}
        for _tree_parameter_name, _tree_bounds in corruption_params[_tree_corruption_name].items():
            _tree_display_name = _tree_parameter_name[:-2] if _tree_parameter_name.endswith(("_r", "_c")) else _tree_parameter_name
            _tree_parameter_key_map[_tree_display_name] = _tree_parameter_name
            _tree_low_value = float(_tree_bounds[0])
            _tree_high_value = float(_tree_bounds[1])
            _tree_parameter_widget_map[_tree_display_name] = mo.ui.slider(
                start=_tree_low_value,
                stop=_tree_high_value,
                step=max((_tree_high_value - _tree_low_value) / 200.0, 0.0005),
                value=float((_tree_low_value + _tree_high_value) / 2.0),
                show_value=True,
                include_input=True,
            )
        _tree_corruption_widget_map[_tree_corruption_name] = mo.ui.dictionary(_tree_parameter_widget_map, label="parameters")
        _tree_corruption_key_maps[_tree_corruption_name] = _tree_parameter_key_map
    tree_corruption_key_maps = _tree_corruption_key_maps
    tree_corruption_controls = mo.ui.dictionary(_tree_corruption_widget_map)
    tree_corruption_widget_map = _tree_corruption_widget_map
    return tree_corruption_controls, tree_corruption_key_maps, tree_corruption_widget_map


@app.cell
def _(
    add_to_mask,
    add_to_pattern,
    classes,
    combine_signals,
    full_ellipse,
    init_output_dictionary,
    inspect,
    mask_sample,
    np,
    plt,
    tree_height_widget,
    tree_seed_widget,
    tree_width_widget,
):
    def tree_call_with_optional_rng(_tree_function, *_tree_args, _tree_seed: int, **_tree_kwargs):
        if "rng" in inspect.signature(_tree_function).parameters:
            _tree_kwargs["rng"] = np.random.default_rng(_tree_seed)
        return _tree_function(*_tree_args, **_tree_kwargs)

    def tree_build_pattern_stage_sample(
        _tree_background_array: np.ndarray,
        _tree_temp_array: np.ndarray,
        _tree_pattern_values: dict,
        _tree_class_index: int,
        _tree_shape: tuple[int, int],
    ) -> dict:
        _tree_stage_sample = init_output_dictionary(None, size=_tree_shape, num_classes=len(classes))
        _tree_stage_sample["background"] = np.array(_tree_background_array, copy=True)
        _tree_stage_sample["temp"] = np.array(_tree_temp_array, copy=True)
        _tree_stage_sample = add_to_mask(_tree_stage_sample, idx=_tree_class_index)
        _tree_stage_sample = add_to_pattern(
            _tree_stage_sample,
            polarity=_tree_pattern_values.get("polarity", "+"),
            value=float(_tree_pattern_values.get("value", 1.0)),
        )
        _tree_stage_sample = combine_signals(_tree_stage_sample)
        _tree_stage_sample = mask_sample(_tree_stage_sample)
        return _tree_stage_sample

    _tree_shape = (int(tree_height_widget.value), int(tree_width_widget.value))
    _tree_wafer_mask = full_ellipse(_tree_shape)
    _tree_seed = int(tree_seed_widget.value)
    _tree_transparent_nan_cmap = plt.get_cmap("jet").copy()
    _tree_transparent_nan_cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    tree_seed = _tree_seed
    tree_shape = _tree_shape
    tree_transparent_nan_cmap = _tree_transparent_nan_cmap
    tree_wafer_mask = _tree_wafer_mask
    return (
        tree_build_pattern_stage_sample,
        tree_call_with_optional_rng,
        tree_seed,
        tree_shape,
        tree_transparent_nan_cmap,
        tree_wafer_mask,
    )


@app.cell
def _(tree_background_widget_map):
    tree_affine_controls = tree_background_widget_map["affine"]
    tree_beta_weibull_controls = tree_background_widget_map["beta_weibull"]
    tree_gaussian_filter_controls = tree_background_widget_map["gaussian_filter"]
    tree_impulse_noise_controls = tree_background_widget_map["impulse_noise"]
    tree_linear_gradient_controls = tree_background_widget_map["linear_gradient"]
    tree_linear_radial_gradient_controls = tree_background_widget_map["linear_radial_gradient"]
    tree_pink_noise_controls = tree_background_widget_map["pink_noise"]
    return (
        tree_affine_controls,
        tree_beta_weibull_controls,
        tree_gaussian_filter_controls,
        tree_impulse_noise_controls,
        tree_linear_gradient_controls,
        tree_linear_radial_gradient_controls,
        tree_pink_noise_controls,
    )


@app.cell
def _(
    beta_weibull,
    compact_heatmap,
    mo,
    np,
    plt,
    tree_background_key_maps,
    tree_beta_weibull_controls,
    tree_call_with_optional_rng,
    tree_save_plots_directory,
    tree_save_plots_enabled,
    tree_seed,
    tree_shape,
    tree_transparent_nan_cmap,
    tree_wafer_mask,
    widget_values_to_kwargs,
):
    tree_beta_weibull_kwargs = widget_values_to_kwargs(
        tree_beta_weibull_controls.value,
        tree_background_key_maps["beta_weibull"],
    )
    tree_beta_weibull_output = np.asarray(
        tree_call_with_optional_rng(
            beta_weibull,
            _tree_seed=tree_seed + 1,
            size=tree_shape,
            **tree_beta_weibull_kwargs,
        ),
        dtype=np.float32,
    )
    tree_beta_weibull_masked = np.where(tree_wafer_mask == 0.0, np.nan, tree_beta_weibull_output)
    tree_beta_weibull_figure, tree_beta_weibull_axis = plt.subplots(figsize=(3.0, 3.0))
    compact_heatmap(tree_beta_weibull_masked, ax=tree_beta_weibull_axis, title=None, cmap=tree_transparent_nan_cmap)
    tree_beta_weibull_figure.patch.set_alpha(0.0)
    tree_beta_weibull_axis.set_facecolor((1.0, 1.0, 1.0, 0.0))
    tree_beta_weibull_figure.tight_layout()
    if tree_save_plots_enabled and tree_save_plots_directory is not None:
        tree_beta_weibull_figure.savefig(tree_save_plots_directory / "01_beta_weibull.png", bbox_inches="tight", dpi=200, transparent=True)
    tree_beta_weibull_widget = mo.ui.matplotlib(tree_beta_weibull_axis)
    plt.close(tree_beta_weibull_figure)
    mo.vstack(
        [
            mo.md("#### 🔔 Beta Weibull"),
            mo.hstack([tree_beta_weibull_controls, tree_beta_weibull_widget], gap=2, widths=[1, 1], align="start"),
        ],
        gap=1,
    )
    return (tree_beta_weibull_output,)


@app.cell
def _(
    compact_heatmap,
    mo,
    np,
    pink_noise,
    plt,
    tree_background_key_maps,
    tree_beta_weibull_output,
    tree_call_with_optional_rng,
    tree_pink_noise_controls,
    tree_save_plots_directory,
    tree_save_plots_enabled,
    tree_seed,
    tree_shape,
    tree_transparent_nan_cmap,
    tree_wafer_mask,
    widget_values_to_kwargs,
):
    tree_pink_noise_kwargs = widget_values_to_kwargs(
        tree_pink_noise_controls.value,
        tree_background_key_maps["pink_noise"],
    )
    tree_pink_noise_output = tree_beta_weibull_output + np.asarray(
        tree_call_with_optional_rng(
            pink_noise,
            _tree_seed=tree_seed + 2,
            size=tree_shape,
            **tree_pink_noise_kwargs,
        ),
        dtype=np.float32,
    )
    tree_pink_noise_masked = np.where(tree_wafer_mask == 0.0, np.nan, tree_pink_noise_output)
    tree_pink_noise_figure, tree_pink_noise_axis = plt.subplots(figsize=(3.0, 3.0))
    compact_heatmap(tree_pink_noise_masked, ax=tree_pink_noise_axis, title=None, cmap=tree_transparent_nan_cmap)
    tree_pink_noise_figure.patch.set_alpha(0.0)
    tree_pink_noise_axis.set_facecolor((1.0, 1.0, 1.0, 0.0))
    tree_pink_noise_figure.tight_layout()
    if tree_save_plots_enabled and tree_save_plots_directory is not None:
        tree_pink_noise_figure.savefig(tree_save_plots_directory / "02_pink_noise.png", bbox_inches="tight", dpi=200, transparent=True)
    tree_pink_noise_widget = mo.ui.matplotlib(tree_pink_noise_axis)
    plt.close(tree_pink_noise_figure)
    mo.vstack(
        [
            mo.md("#### 🪷 Pink Noise"),
            mo.hstack([tree_pink_noise_controls, tree_pink_noise_widget], gap=2, widths=[1, 1], align="start"),
        ],
        gap=1,
    )
    return (tree_pink_noise_output,)


@app.cell
def _(
    compact_heatmap,
    gaussian_filter,
    mo,
    np,
    plt,
    tree_background_key_maps,
    tree_gaussian_filter_controls,
    tree_pink_noise_output,
    tree_save_plots_directory,
    tree_save_plots_enabled,
    tree_transparent_nan_cmap,
    tree_wafer_mask,
    widget_values_to_kwargs,
):
    tree_gaussian_filter_kwargs = widget_values_to_kwargs(
        tree_gaussian_filter_controls.value,
        tree_background_key_maps["gaussian_filter"],
    )
    tree_gaussian_filter_output = np.asarray(
        gaussian_filter(np.asarray(tree_pink_noise_output, dtype=np.float32), **tree_gaussian_filter_kwargs),
        dtype=np.float32,
    )
    tree_gaussian_filter_masked = np.where(tree_wafer_mask == 0.0, np.nan, tree_gaussian_filter_output)
    tree_gaussian_filter_figure, tree_gaussian_filter_axis = plt.subplots(figsize=(3.0, 3.0))
    compact_heatmap(tree_gaussian_filter_masked, ax=tree_gaussian_filter_axis, title=None, cmap=tree_transparent_nan_cmap)
    tree_gaussian_filter_figure.patch.set_alpha(0.0)
    tree_gaussian_filter_axis.set_facecolor((1.0, 1.0, 1.0, 0.0))
    tree_gaussian_filter_figure.tight_layout()
    if tree_save_plots_enabled and tree_save_plots_directory is not None:
        tree_gaussian_filter_figure.savefig(
            tree_save_plots_directory / "03_gaussian_filter.png", bbox_inches="tight", dpi=200, transparent=True
        )
    tree_gaussian_filter_widget = mo.ui.matplotlib(tree_gaussian_filter_axis)
    plt.close(tree_gaussian_filter_figure)
    mo.vstack(
        [
            mo.md("#### ☁️ Gaussian Filter"),
            mo.hstack([tree_gaussian_filter_controls, tree_gaussian_filter_widget], gap=2, widths=[1, 1], align="start"),
        ],
        gap=1,
    )
    return (tree_gaussian_filter_output,)


@app.cell
def _(
    compact_heatmap,
    impulse_noise,
    mo,
    np,
    plt,
    tree_background_key_maps,
    tree_call_with_optional_rng,
    tree_gaussian_filter_output,
    tree_impulse_noise_controls,
    tree_save_plots_directory,
    tree_save_plots_enabled,
    tree_seed,
    tree_shape,
    tree_transparent_nan_cmap,
    tree_wafer_mask,
    widget_values_to_kwargs,
):
    tree_impulse_noise_kwargs = widget_values_to_kwargs(
        tree_impulse_noise_controls.value,
        tree_background_key_maps["impulse_noise"],
    )
    tree_impulse_noise_output = tree_gaussian_filter_output + np.asarray(
        tree_call_with_optional_rng(
            impulse_noise,
            _tree_seed=tree_seed + 3,
            size=tree_shape,
            **tree_impulse_noise_kwargs,
        ),
        dtype=np.float32,
    )
    tree_impulse_noise_masked = np.where(tree_wafer_mask == 0.0, np.nan, tree_impulse_noise_output)
    tree_impulse_noise_figure, tree_impulse_noise_axis = plt.subplots(figsize=(3.0, 3.0))
    compact_heatmap(tree_impulse_noise_masked, ax=tree_impulse_noise_axis, title=None, cmap=tree_transparent_nan_cmap)
    tree_impulse_noise_figure.patch.set_alpha(0.0)
    tree_impulse_noise_axis.set_facecolor((1.0, 1.0, 1.0, 0.0))
    tree_impulse_noise_figure.tight_layout()
    if tree_save_plots_enabled and tree_save_plots_directory is not None:
        tree_impulse_noise_figure.savefig(
            tree_save_plots_directory / "04_impulse_noise.png", bbox_inches="tight", dpi=200, transparent=True
        )
    tree_impulse_noise_widget = mo.ui.matplotlib(tree_impulse_noise_axis)
    plt.close(tree_impulse_noise_figure)
    mo.vstack(
        [
            mo.md("#### 🔨 Impulse Noise"),
            mo.hstack([tree_impulse_noise_controls, tree_impulse_noise_widget], gap=2, widths=[1, 1], align="start"),
        ],
        gap=1,
    )
    return (tree_impulse_noise_output,)


@app.cell
def _(
    compact_heatmap,
    linear_gradient,
    mo,
    np,
    plt,
    tree_background_key_maps,
    tree_impulse_noise_output,
    tree_linear_gradient_controls,
    tree_save_plots_directory,
    tree_save_plots_enabled,
    tree_shape,
    tree_transparent_nan_cmap,
    tree_wafer_mask,
    widget_values_to_kwargs,
):
    tree_linear_gradient_kwargs = widget_values_to_kwargs(
        tree_linear_gradient_controls.value,
        tree_background_key_maps["linear_gradient"],
    )
    tree_linear_gradient_output = tree_impulse_noise_output * np.asarray(
        linear_gradient(shape=tree_shape, **tree_linear_gradient_kwargs),
        dtype=np.float32,
    )
    tree_linear_gradient_masked = np.where(tree_wafer_mask == 0.0, np.nan, tree_linear_gradient_output)
    tree_linear_gradient_figure, tree_linear_gradient_axis = plt.subplots(figsize=(3.0, 3.0))
    compact_heatmap(tree_linear_gradient_masked, ax=tree_linear_gradient_axis, title=None, cmap=tree_transparent_nan_cmap)
    tree_linear_gradient_figure.patch.set_alpha(0.0)
    tree_linear_gradient_axis.set_facecolor((1.0, 1.0, 1.0, 0.0))
    tree_linear_gradient_figure.tight_layout()
    if tree_save_plots_enabled and tree_save_plots_directory is not None:
        tree_linear_gradient_figure.savefig(
            tree_save_plots_directory / "05_linear_gradient.png", bbox_inches="tight", dpi=200, transparent=True
        )
    tree_linear_gradient_widget = mo.ui.matplotlib(tree_linear_gradient_axis)
    plt.close(tree_linear_gradient_figure)
    mo.vstack(
        [
            mo.md("#### 📏 Linear Gradient"),
            mo.hstack([tree_linear_gradient_controls, tree_linear_gradient_widget], gap=2, widths=[1, 1], align="start"),
        ],
        gap=1,
    )
    return (tree_linear_gradient_output,)


@app.cell
def _(
    compact_heatmap,
    linear_radial_gradient,
    mo,
    np,
    plt,
    tree_background_key_maps,
    tree_linear_gradient_output,
    tree_linear_radial_gradient_controls,
    tree_save_plots_directory,
    tree_save_plots_enabled,
    tree_shape,
    tree_transparent_nan_cmap,
    tree_wafer_mask,
    widget_values_to_kwargs,
):
    tree_linear_radial_gradient_kwargs = widget_values_to_kwargs(
        tree_linear_radial_gradient_controls.value,
        tree_background_key_maps["linear_radial_gradient"],
    )
    tree_linear_radial_gradient_output = tree_linear_gradient_output * np.asarray(
        linear_radial_gradient(shape=tree_shape, **tree_linear_radial_gradient_kwargs),
        dtype=np.float32,
    )
    tree_linear_radial_gradient_masked = np.where(tree_wafer_mask == 0.0, np.nan, tree_linear_radial_gradient_output)
    tree_linear_radial_gradient_figure, tree_linear_radial_gradient_axis = plt.subplots(figsize=(3.0, 3.0))
    compact_heatmap(
        tree_linear_radial_gradient_masked,
        ax=tree_linear_radial_gradient_axis,
        title=None,
        cmap=tree_transparent_nan_cmap,
    )
    tree_linear_radial_gradient_figure.patch.set_alpha(0.0)
    tree_linear_radial_gradient_axis.set_facecolor((1.0, 1.0, 1.0, 0.0))
    tree_linear_radial_gradient_figure.tight_layout()
    if tree_save_plots_enabled and tree_save_plots_directory is not None:
        tree_linear_radial_gradient_figure.savefig(
            tree_save_plots_directory / "06_linear_radial_gradient.png", bbox_inches="tight", dpi=200, transparent=True
        )
    tree_linear_radial_gradient_widget = mo.ui.matplotlib(tree_linear_radial_gradient_axis)
    plt.close(tree_linear_radial_gradient_figure)
    mo.vstack(
        [
            mo.md("#### 🛞 Linear Radial Gradient"),
            mo.hstack([tree_linear_radial_gradient_controls, tree_linear_radial_gradient_widget], gap=2, widths=[1, 1], align="start"),
        ],
        gap=1,
    )
    return (tree_linear_radial_gradient_output,)


@app.cell
def _(
    affine,
    compact_heatmap,
    mo,
    np,
    plt,
    tree_affine_controls,
    tree_background_key_maps,
    tree_linear_radial_gradient_output,
    tree_save_plots_directory,
    tree_save_plots_enabled,
    tree_transparent_nan_cmap,
    tree_wafer_mask,
    widget_values_to_kwargs,
):
    tree_affine_kwargs = widget_values_to_kwargs(
        tree_affine_controls.value,
        tree_background_key_maps["affine"],
    )
    tree_affine_output = np.asarray(
        affine(np.asarray(tree_linear_radial_gradient_output, dtype=np.float32), **tree_affine_kwargs),
        dtype=np.float32,
    )
    tree_affine_masked = np.where(tree_wafer_mask == 0.0, np.nan, tree_affine_output)
    tree_affine_figure, tree_affine_axis = plt.subplots(figsize=(3.0, 3.0))
    compact_heatmap(tree_affine_masked, ax=tree_affine_axis, title=None, cmap=tree_transparent_nan_cmap)
    tree_affine_figure.patch.set_alpha(0.0)
    tree_affine_axis.set_facecolor((1.0, 1.0, 1.0, 0.0))
    tree_affine_figure.tight_layout()
    if tree_save_plots_enabled and tree_save_plots_directory is not None:
        tree_affine_figure.savefig(tree_save_plots_directory / "07_affine.png", bbox_inches="tight", dpi=200, transparent=True)
    tree_affine_widget = mo.ui.matplotlib(tree_affine_axis)
    plt.close(tree_affine_figure)
    mo.vstack(
        [
            mo.md("#### 📐 Affine"),
            mo.hstack([tree_affine_controls, tree_affine_widget], gap=2, widths=[1, 1], align="start"),
        ],
        gap=1,
    )
    return (tree_affine_output,)


@app.cell
def _(
    build_pattern_corruption_kwargs,
    classes,
    np,
    pattern_corruptions,
    pattern_generators,
    tree_affine_output,
    tree_build_pattern_stage_sample,
    tree_call_with_optional_rng,
    tree_corruption_key_maps,
    tree_corruption_widget_map,
    tree_pattern_parameter_key_map,
    tree_pattern_parameter_widget,
    tree_seed,
    tree_shape,
    tree_single_class_widget,
    widget_values_to_kwargs,
):
    _tree_class_name = tree_single_class_widget.value
    _tree_class_index = classes.index(_tree_class_name)
    _tree_pattern_values = tree_pattern_parameter_widget.value
    _tree_pattern_widget_kwargs = widget_values_to_kwargs(
        _tree_pattern_values,
        tree_pattern_parameter_key_map,
        skip_parameters={"value_r", "polarity_c"},
    )
    _tree_pattern_kwargs = build_pattern_generator_kwargs(
        _tree_class_name,
        _tree_pattern_widget_kwargs,
        shape=tree_shape,
        skip_parameters={"value_r", "polarity_c"},
    )

    _tree_temp_array = np.asarray(
        tree_call_with_optional_rng(
            pattern_generators[_tree_class_name],
            np.zeros(tree_shape, dtype=np.float32),
            _tree_seed=tree_seed + 10,
            **_tree_pattern_kwargs,
        ),
        dtype=np.float32,
    )
    _tree_generator_sample = tree_build_pattern_stage_sample(
        tree_affine_output,
        _tree_temp_array,
        _tree_pattern_values,
        _tree_class_index,
        tree_shape,
    )

    _tree_pattern_sections = [
        (
            "🌘",
            _tree_class_name.replace("_", " ").title(),
            tree_pattern_parameter_widget,
            np.asarray(_tree_generator_sample["sample"], dtype=np.float32),
            np.asarray(_tree_generator_sample["masks"], dtype=np.float32),
        )
    ]

    _tree_final_sample = _tree_generator_sample
    for _tree_corruption_index, _tree_corruption_function in enumerate(pattern_corruptions[_tree_class_name], start=1):
        _tree_corruption_name = _tree_corruption_function.__name__
        _tree_corruption_values = tree_corruption_widget_map[_tree_corruption_name].value
        _tree_corruption_kwargs = widget_values_to_kwargs(
            _tree_corruption_values,
            tree_corruption_key_maps[_tree_corruption_name],
        )
        _tree_corruption_kwargs = build_pattern_corruption_kwargs(
            _tree_corruption_function,
            _tree_corruption_kwargs,
            shape=tree_shape,
        )
        _tree_temp_array = np.asarray(
            tree_call_with_optional_rng(
                _tree_corruption_function,
                np.array(_tree_temp_array, copy=True),
                _tree_seed=tree_seed + 100 + _tree_corruption_index,
                **_tree_corruption_kwargs,
            ),
            dtype=np.float32,
        )
        _tree_final_sample = tree_build_pattern_stage_sample(
            tree_affine_output,
            _tree_temp_array,
            _tree_pattern_values,
            _tree_class_index,
            tree_shape,
        )
        _tree_pattern_sections.append(
            (
                "🔨",
                _tree_corruption_name.replace("_", " ").title(),
                tree_corruption_widget_map[_tree_corruption_name],
                np.asarray(_tree_final_sample["sample"], dtype=np.float32),
                np.asarray(_tree_final_sample["masks"], dtype=np.float32),
            )
        )

    tree_final_sample = _tree_final_sample
    tree_pattern_sections = _tree_pattern_sections
    return tree_final_sample, tree_pattern_sections


@app.cell
def _(
    compact_heatmap,
    mo,
    plot_mask,
    plt,
    tree_pattern_sections,
    tree_single_class_widget,
    tree_save_plots_directory,
    tree_save_plots_enabled,
    tree_transparent_nan_cmap,
):
    _tree_pattern_blocks = [mo.md("### Pattern branch")]
    _tree_pattern_name = str(tree_single_class_widget.value)
    _tree_safe_pattern_name = "".join(
        _tree_character if _tree_character.isalnum() or _tree_character in "-_" else "_" for _tree_character in _tree_pattern_name
    )
    for _tree_section_emoji, _tree_section_name, _tree_section_controls, _tree_section_sample, _tree_section_masks in tree_pattern_sections:
        _tree_sample_figure, _tree_sample_axis = plt.subplots(figsize=(3.0, 3.0))
        compact_heatmap(_tree_section_sample, ax=_tree_sample_axis, title=None, cmap=tree_transparent_nan_cmap)
        _tree_sample_figure.patch.set_alpha(0.0)
        _tree_sample_axis.set_facecolor((1.0, 1.0, 1.0, 0.0))
        _tree_sample_figure.tight_layout()
        _tree_safe_section_name = "".join(
            _tree_character if _tree_character.isalnum() or _tree_character in "-_" else "_" for _tree_character in _tree_section_name
        )
        _tree_sample_filename = f"08_{tree_single_class_widget.value}_{_tree_safe_section_name}.png"
        if tree_save_plots_enabled and tree_save_plots_directory is not None:
            _tree_sample_figure.savefig(
                tree_save_plots_directory / _tree_sample_filename,
                bbox_inches="tight",
                dpi=200,
                transparent=True,
            )
        _tree_sample_widget = mo.ui.matplotlib(_tree_sample_axis)
        plt.close(_tree_sample_figure)

        _tree_mask_figure, _tree_mask_axis = plt.subplots(figsize=(3.0, 3.0))
        plot_mask(_tree_section_masks, create_fig=False, ax=_tree_mask_axis, legend=False)
        _tree_mask_figure.tight_layout()
        _tree_mask_widget = mo.ui.matplotlib(_tree_mask_axis)
        plt.close(_tree_mask_figure)

        _tree_pattern_blocks.extend(
            [
                mo.md(f"#### {_tree_section_emoji} {_tree_section_name}"),
                mo.hstack(
                    [_tree_section_controls, _tree_sample_widget, _tree_mask_widget],
                    gap=2,
                    widths=[1, 1, 1],
                    align="start",
                ),
            ]
        )
    mo.vstack(_tree_pattern_blocks, gap=1)
    return


@app.cell
def _(
    compact_heatmap,
    mo,
    np,
    plot_compact_sinogram,
    plot_ecdf,
    plot_fft,
    plot_histogram,
    plt,
    tree_final_sample,
    tree_transparent_nan_cmap,
):
    _tree_analysis_signal = np.asarray(tree_final_sample["sample"], dtype=np.float32)
    _tree_analysis_figure, _tree_analysis_axes = plt.subplots(1, 5, figsize=(16, 3), layout="constrained")
    compact_heatmap(_tree_analysis_signal, ax=_tree_analysis_axes[0], title=None, cmap=tree_transparent_nan_cmap)
    _tree_analysis_figure.patch.set_alpha(0.0)
    _tree_analysis_axes[0].set_facecolor((1.0, 1.0, 1.0, 0.0))
    plot_fft(_tree_analysis_signal, ax=_tree_analysis_axes[1], create_fig=False)
    plot_compact_sinogram(_tree_analysis_signal, ax=_tree_analysis_axes[2], create_fig=False)
    plot_ecdf(_tree_analysis_signal, ax=_tree_analysis_axes[3], title="ECDF", create_fig=False)
    plot_histogram(_tree_analysis_signal, ax=_tree_analysis_axes[4], title="Histogram", create_fig=False)
    _tree_analysis_widget = mo.ui.matplotlib(_tree_analysis_axes[0])
    plt.close(_tree_analysis_figure)
    _tree_analysis_widget
    return


@app.cell
def _(background_params, corruption_params, mo, pattern_params):
    _tree_gallery_background_widget_map = {}
    _tree_gallery_branch_widget_map = {}
    _tree_gallery_key_maps = {"background": {}, "branches": {}}

    for _tree_node_name, _tree_parameter_bounds in background_params.items():
        _tree_parameter_widget_map = {}
        _tree_parameter_key_map = {}
        for _tree_parameter_name, _tree_bounds in _tree_parameter_bounds.items():
            _tree_display_name = _tree_parameter_name[:-2] if _tree_parameter_name.endswith(("_r", "_c")) else _tree_parameter_name
            _tree_parameter_key_map[_tree_display_name] = _tree_parameter_name
            if _tree_parameter_name.endswith("_r"):
                _tree_low_value = float(_tree_bounds[0])
                _tree_high_value = float(_tree_bounds[1])
                _tree_parameter_widget_map[_tree_display_name] = mo.ui.range_slider(
                    start=_tree_low_value,
                    stop=_tree_high_value,
                    step=max((_tree_high_value - _tree_low_value) / 200.0, 0.0005),
                    value=(_tree_low_value, _tree_high_value),
                    show_value=True,
                )
        _tree_gallery_background_widget_map[_tree_node_name] = mo.ui.dictionary(_tree_parameter_widget_map)
        _tree_gallery_key_maps["background"][_tree_node_name] = _tree_parameter_key_map

    for _tree_entry_name, _tree_parameter_bounds in (pattern_params | corruption_params).items():
        if not isinstance(_tree_parameter_bounds, dict):
            continue
        _tree_parameter_widget_map = {}
        _tree_parameter_key_map = {}
        for _tree_parameter_name, _tree_bounds in _tree_parameter_bounds.items():
            _tree_display_name = _tree_parameter_name[:-2] if _tree_parameter_name.endswith(("_r", "_c")) else _tree_parameter_name
            _tree_parameter_key_map[_tree_display_name] = _tree_parameter_name
            if _tree_parameter_name.endswith("_r"):
                _tree_low_value = float(_tree_bounds[0])
                _tree_high_value = float(_tree_bounds[1])
                _tree_parameter_widget_map[_tree_display_name] = mo.ui.range_slider(
                    start=_tree_low_value,
                    stop=_tree_high_value,
                    step=max((_tree_high_value - _tree_low_value) / 200.0, 0.0005),
                    value=(_tree_low_value, _tree_high_value),
                    show_value=True,
                )
            elif _tree_parameter_name.endswith("_c"):
                _tree_initial_option_name = str(_tree_bounds[0]).lower() if isinstance(_tree_bounds[0], bool) else str(_tree_bounds[0])
                _tree_parameter_widget_map[_tree_display_name] = mo.ui.dropdown(
                    {
                        str(_tree_choice).lower() if isinstance(_tree_choice, bool) else str(_tree_choice): _tree_choice
                        for _tree_choice in _tree_bounds
                    },
                    value=_tree_initial_option_name,
                )
        _tree_gallery_branch_widget_map[_tree_entry_name] = mo.ui.dictionary(_tree_parameter_widget_map)
        _tree_gallery_key_maps["branches"][_tree_entry_name] = _tree_parameter_key_map

    tree_gallery_key_maps = _tree_gallery_key_maps
    tree_gallery_ranges_widget = mo.ui.dictionary(
        {
            "background": mo.ui.dictionary(_tree_gallery_background_widget_map),
            "branches": mo.ui.dictionary(_tree_gallery_branch_widget_map),
        }
    )
    return tree_gallery_key_maps, tree_gallery_ranges_widget


@app.cell
def _(list_available_components, mo):
    _tree_cardinality_prior_names = list_available_components("cardinality_priors")
    _tree_class_prior_names = list_available_components("class_priors")
    _tree_default_cardinality_prior = "uniform" if "uniform" in _tree_cardinality_prior_names else _tree_cardinality_prior_names[0]
    _tree_default_class_prior = "uniform" if "uniform" in _tree_class_prior_names else _tree_class_prior_names[0]
    tree_gallery_cardinality_prior_widget = mo.ui.dropdown(
        {name: name for name in _tree_cardinality_prior_names},
        value=_tree_default_cardinality_prior,
        label="cardinality prior",
    )
    tree_gallery_class_prior_widget = mo.ui.dropdown(
        {name: name for name in _tree_class_prior_names},
        value=_tree_default_class_prior,
        label="class prior",
    )
    return (
        tree_gallery_cardinality_prior_widget,
        tree_gallery_class_prior_widget,
    )


@app.cell
def _(
    ConceptDriftConfig,
    CovariateDriftConfig,
    DriftConfig,
    PriorDriftConfig,
    TreeFactory,
    load_component_config,
    tree_gallery_cardinality_prior_widget,
    tree_gallery_class_prior_widget,
    tree_gallery_key_maps,
    tree_gallery_ranges_widget,
    tree_height_widget,
    tree_seed_widget,
    tree_width_widget,
):
    _tree_gallery_stem_config = load_component_config("stem", "full")
    _tree_gallery_branches_config = load_component_config("branches", "multi_label")
    _tree_gallery_cardinality_prior_config = load_component_config("cardinality_priors", tree_gallery_cardinality_prior_widget.value)
    _tree_gallery_class_prior_config = load_component_config("class_priors", tree_gallery_class_prior_widget.value)
    _tree_gallery_postprocessing_config = load_component_config("postprocessing", "continuous")
    _tree_gallery_shape_config = load_component_config("shapes", "fixed")
    _tree_gallery_shape_config.params["width"] = int(tree_width_widget.value)
    _tree_gallery_shape_config.params["height"] = int(tree_height_widget.value)

    _tree_gallery_values = tree_gallery_ranges_widget.value
    for _tree_node_name, _tree_parameter_values in _tree_gallery_values["background"].items():
        _tree_key_map = tree_gallery_key_maps["background"][_tree_node_name]
        for _tree_display_name, _tree_widget_value in _tree_parameter_values.items():
            _tree_parameter_name = _tree_key_map[_tree_display_name]
            if _tree_parameter_name.endswith("_r"):
                _tree_gallery_stem_config.params[_tree_node_name][_tree_parameter_name] = (
                    float(_tree_widget_value[0]),
                    float(_tree_widget_value[1]),
                )
            elif _tree_parameter_name.endswith("_c"):
                _tree_gallery_stem_config.params[_tree_node_name][_tree_parameter_name] = (_tree_widget_value,)

    for _tree_entry_name, _tree_parameter_values in _tree_gallery_values["branches"].items():
        _tree_key_map = tree_gallery_key_maps["branches"][_tree_entry_name]
        for _tree_display_name, _tree_widget_value in _tree_parameter_values.items():
            _tree_parameter_name = _tree_key_map[_tree_display_name]
            if _tree_parameter_name.endswith("_r"):
                _tree_gallery_branches_config.params[_tree_entry_name][_tree_parameter_name] = (
                    float(_tree_widget_value[0]),
                    float(_tree_widget_value[1]),
                )
            elif _tree_parameter_name.endswith("_c"):
                _tree_gallery_branches_config.params[_tree_entry_name][_tree_parameter_name] = (_tree_widget_value,)

    _tree_gallery_drift_config = DriftConfig(
        covariate=CovariateDriftConfig(strategy="fixed_full", params={}),
        concept=ConceptDriftConfig(strategy="fixed_full", params={}),
        prior=PriorDriftConfig(cardinality_schedule="fixed", class_schedule="fixed", params={}),
    )

    tree_gallery_sampling_tree = TreeFactory.from_configs(
        stem_config=_tree_gallery_stem_config,
        branches_config=_tree_gallery_branches_config,
        cardinality_prior_config=_tree_gallery_cardinality_prior_config,
        class_prior_config=_tree_gallery_class_prior_config,
        postprocessing_config=_tree_gallery_postprocessing_config,
        shape_config=_tree_gallery_shape_config,
        drift_config=_tree_gallery_drift_config,
        num_tasks=1,
        seed=int(tree_seed_widget.value),
    )
    return (tree_gallery_sampling_tree,)


@app.cell
def _(
    compact_heatmap,
    mo,
    plot_mask,
    plt,
    tree_gallery_cardinality_prior_widget,
    tree_gallery_class_prior_widget,
    tree_gallery_ranges_widget,
    tree_gallery_sampling_tree,
):
    _tree_signal_widgets = []
    _tree_mask_widgets = []

    for _tree_gallery_index in range(12):
        _tree_gallery_sample = tree_gallery_sampling_tree.sample()

        _tree_signal_figure, _tree_signal_axis = plt.subplots(figsize=(2.4, 2.4))
        compact_heatmap(_tree_gallery_sample["sample"], ax=_tree_signal_axis)
        _tree_signal_figure.tight_layout()
        _tree_signal_widgets.append(mo.ui.matplotlib(_tree_signal_axis))
        plt.close(_tree_signal_figure)

        _tree_mask_figure, _tree_mask_axis = plt.subplots(figsize=(2.4, 2.4))
        plot_mask(_tree_gallery_sample["masks"], create_fig=False, ax=_tree_mask_axis, legend=False)
        _tree_mask_figure.tight_layout()
        _tree_mask_widgets.append(mo.ui.matplotlib(_tree_mask_axis))
        plt.close(_tree_mask_figure)

    _tree_signal_rows = [
        mo.hstack(_tree_signal_widgets[_tree_row_index : _tree_row_index + 4], justify="start", gap=1)
        for _tree_row_index in range(0, len(_tree_signal_widgets), 4)
    ]
    _tree_mask_rows = [
        mo.hstack(_tree_mask_widgets[_tree_row_index : _tree_row_index + 4], justify="start", gap=1)
        for _tree_row_index in range(0, len(_tree_mask_widgets), 4)
    ]
    _tree_gallery_tabs = mo.ui.tabs(
        {
            "Signal": mo.vstack(_tree_signal_rows, gap=1),
            "Masks": mo.vstack(_tree_mask_rows, gap=1),
        },
        value="Signal",
    )

    mo.vstack(
        [
            mo.md("## 🖼️ Gallery"),
            mo.hstack(
                [tree_gallery_cardinality_prior_widget, tree_gallery_class_prior_widget],
                justify="start",
                gap=1,
                wrap=True,
            ),
            tree_gallery_ranges_widget,
            _tree_gallery_tabs,
        ],
        gap=1,
    )
    return


if __name__ == "__main__":
    app.run()
