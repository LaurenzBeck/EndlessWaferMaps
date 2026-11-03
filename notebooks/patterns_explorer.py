# ruff: noqa: ANN, B018, E402, INP001, I001

"""🌘 Marimo patterns explorer."""

import marimo

__generated_with = "0.22.0"
app = marimo.App(width="full", app_title="🌘 Patterns Explorer")


@app.cell
def _():
    import inspect

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from endlesswm.explorer_support import build_pattern_corruption_kwargs, build_pattern_generator_kwargs
    from endlesswm.geometry import full_ellipse
    from endlesswm.patterns import (
        add_to_mask,
        add_to_pattern,
        classes,
        corruption_params,
        pattern2idx,
        pattern_corruptions,
        pattern_generators,
        pattern_params,
    )
    from endlesswm.trees import combine_signals, init_output_dictionary, mask_sample
    from endlesswm.visualizations import compact_heatmap, plot_compact_sinogram, plot_ecdf, plot_fft, plot_histogram, plot_mask

    return (
        add_to_mask,
        add_to_pattern,
        build_pattern_corruption_kwargs,
        build_pattern_generator_kwargs,
        classes,
        combine_signals,
        compact_heatmap,
        corruption_params,
        full_ellipse,
        init_output_dictionary,
        inspect,
        mask_sample,
        mo,
        np,
        pattern2idx,
        pattern_corruptions,
        pattern_generators,
        pattern_params,
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
    # 🌘 EndlessWM - Patterns Explorer
    """)
    return


@app.cell
def _(classes, mo):
    patterns_height_widget = mo.ui.number(start=16, stop=256, step=1, value=52, label="height")
    patterns_width_widget = mo.ui.number(start=16, stop=256, step=1, value=52, label="width")
    patterns_name_widget = mo.ui.dropdown(
        {class_name: class_name for class_name in classes},
        value="center_spot",
        label="pattern",
    )
    patterns_seed_widget = mo.ui.number(start=0, stop=100000, step=1, value=42, label="seed")
    _patterns_top_section = mo.vstack(
        [
            mo.md("## 📐 Wafer Map Size"),
            mo.hstack([patterns_height_widget, patterns_width_widget], justify="start", gap=1),
            mo.md("## 🌘 Pattern Selection"),
            mo.hstack([patterns_name_widget, patterns_seed_widget], justify="start", gap=1, wrap=True),
        ],
        gap=1,
    )
    _patterns_top_section
    return (
        patterns_height_widget,
        patterns_name_widget,
        patterns_seed_widget,
        patterns_width_widget,
    )


@app.cell
def _(mo, pattern_params, patterns_name_widget):
    _patterns_generator_widget_map = {}
    _patterns_generator_key_map = {}
    for _patterns_parameter_name, _patterns_parameter_bounds in pattern_params[patterns_name_widget.value].items():
        _patterns_display_name = (
            _patterns_parameter_name[:-2] if _patterns_parameter_name.endswith(("_r", "_c")) else _patterns_parameter_name
        )
        _patterns_generator_key_map[_patterns_display_name] = _patterns_parameter_name
        if _patterns_parameter_name.endswith("_r"):
            _patterns_parameter_low_value = float(_patterns_parameter_bounds[0])
            _patterns_parameter_high_value = float(_patterns_parameter_bounds[1])
            _patterns_generator_widget_map[_patterns_display_name] = mo.ui.slider(
                start=_patterns_parameter_low_value,
                stop=_patterns_parameter_high_value,
                step=max((_patterns_parameter_high_value - _patterns_parameter_low_value) / 200.0, 0.0005),
                value=float((_patterns_parameter_low_value + _patterns_parameter_high_value) / 2.0),
                show_value=True,
                include_input=True,
            )
        elif _patterns_parameter_name.endswith("_c"):
            _patterns_initial_option_name = (
                str(_patterns_parameter_bounds[0]).lower()
                if isinstance(_patterns_parameter_bounds[0], bool)
                else str(_patterns_parameter_bounds[0])
            )
            _patterns_generator_widget_map[_patterns_display_name] = mo.ui.dropdown(
                {
                    (
                        str(_patterns_parameter_choice).lower()
                        if isinstance(_patterns_parameter_choice, bool)
                        else str(_patterns_parameter_choice)
                    ): _patterns_parameter_choice
                    for _patterns_parameter_choice in _patterns_parameter_bounds
                },
                value=_patterns_initial_option_name,
            )
    patterns_single_generator_key_map = _patterns_generator_key_map
    patterns_single_generator_widget = mo.ui.dictionary(_patterns_generator_widget_map, label="parameters")
    return patterns_single_generator_key_map, patterns_single_generator_widget


@app.cell
def _(corruption_params, mo, pattern_corruptions, patterns_name_widget):
    def _patterns_build_corruption_stage(_patterns_index: int):
        _patterns_corruption_functions = pattern_corruptions[patterns_name_widget.value]
        if len(_patterns_corruption_functions) <= _patterns_index:
            return None, {}, mo.ui.dictionary({}, label="parameters")

        _patterns_corruption_name = _patterns_corruption_functions[_patterns_index].__name__
        _patterns_corruption_parameter_widget_map = {}
        _patterns_corruption_key_map = {}
        for _patterns_parameter_name, _patterns_parameter_bounds in corruption_params[_patterns_corruption_name].items():
            _patterns_display_name = (
                _patterns_parameter_name[:-2] if _patterns_parameter_name.endswith(("_r", "_c")) else _patterns_parameter_name
            )
            _patterns_corruption_key_map[_patterns_display_name] = _patterns_parameter_name
            _patterns_parameter_low_value = float(_patterns_parameter_bounds[0])
            _patterns_parameter_high_value = float(_patterns_parameter_bounds[1])
            _patterns_corruption_parameter_widget_map[_patterns_display_name] = mo.ui.slider(
                start=_patterns_parameter_low_value,
                stop=_patterns_parameter_high_value,
                step=max((_patterns_parameter_high_value - _patterns_parameter_low_value) / 200.0, 0.0005),
                value=float((_patterns_parameter_low_value + _patterns_parameter_high_value) / 2.0),
                label=_patterns_display_name,
                show_value=True,
                include_input=True,
            )
        return (
            _patterns_corruption_name,
            _patterns_corruption_key_map,
            mo.ui.dictionary(
                _patterns_corruption_parameter_widget_map,
                label="parameters",
            ),
        )

    (
        patterns_single_corruption_stage_1_name,
        patterns_single_corruption_stage_1_key_map,
        patterns_single_corruption_stage_1_controls,
    ) = _patterns_build_corruption_stage(0)
    (
        patterns_single_corruption_stage_2_name,
        patterns_single_corruption_stage_2_key_map,
        patterns_single_corruption_stage_2_controls,
    ) = _patterns_build_corruption_stage(1)
    (
        patterns_single_corruption_stage_3_name,
        patterns_single_corruption_stage_3_key_map,
        patterns_single_corruption_stage_3_controls,
    ) = _patterns_build_corruption_stage(2)
    (
        patterns_single_corruption_stage_4_name,
        patterns_single_corruption_stage_4_key_map,
        patterns_single_corruption_stage_4_controls,
    ) = _patterns_build_corruption_stage(3)
    (
        patterns_single_corruption_stage_5_name,
        patterns_single_corruption_stage_5_key_map,
        patterns_single_corruption_stage_5_controls,
    ) = _patterns_build_corruption_stage(4)
    return (
        patterns_single_corruption_stage_1_controls,
        patterns_single_corruption_stage_1_key_map,
        patterns_single_corruption_stage_1_name,
        patterns_single_corruption_stage_2_controls,
        patterns_single_corruption_stage_2_key_map,
        patterns_single_corruption_stage_2_name,
        patterns_single_corruption_stage_3_controls,
        patterns_single_corruption_stage_3_key_map,
        patterns_single_corruption_stage_3_name,
        patterns_single_corruption_stage_4_controls,
        patterns_single_corruption_stage_4_key_map,
        patterns_single_corruption_stage_4_name,
        patterns_single_corruption_stage_5_controls,
        patterns_single_corruption_stage_5_key_map,
        patterns_single_corruption_stage_5_name,
    )


@app.cell
def _(
    corruption_params,
    mo,
    pattern_corruptions,
    pattern_params,
    patterns_name_widget,
):
    _patterns_gallery_sections = {}
    _patterns_gallery_generator_widget_map = {}
    _patterns_gallery_key_maps = {"generation": {}}

    for _patterns_parameter_name, _patterns_parameter_bounds in pattern_params[patterns_name_widget.value].items():
        _patterns_display_name = (
            _patterns_parameter_name[:-2] if _patterns_parameter_name.endswith(("_r", "_c")) else _patterns_parameter_name
        )
        _patterns_gallery_key_maps["generation"][_patterns_display_name] = _patterns_parameter_name
        if _patterns_parameter_name.endswith("_r"):
            _patterns_parameter_low_value = float(_patterns_parameter_bounds[0])
            _patterns_parameter_high_value = float(_patterns_parameter_bounds[1])
            _patterns_gallery_generator_widget_map[_patterns_display_name] = mo.ui.range_slider(
                start=_patterns_parameter_low_value,
                stop=_patterns_parameter_high_value,
                step=max((_patterns_parameter_high_value - _patterns_parameter_low_value) / 200.0, 0.0005),
                value=(_patterns_parameter_low_value, _patterns_parameter_high_value),
                show_value=True,
            )
        elif _patterns_parameter_name.endswith("_c"):
            _patterns_initial_option_name = (
                str(_patterns_parameter_bounds[0]).lower()
                if isinstance(_patterns_parameter_bounds[0], bool)
                else str(_patterns_parameter_bounds[0])
            )
            _patterns_gallery_generator_widget_map[_patterns_display_name] = mo.ui.dropdown(
                {
                    (
                        str(_patterns_parameter_choice).lower()
                        if isinstance(_patterns_parameter_choice, bool)
                        else str(_patterns_parameter_choice)
                    ): _patterns_parameter_choice
                    for _patterns_parameter_choice in _patterns_parameter_bounds
                },
                value=_patterns_initial_option_name,
            )

    _patterns_gallery_sections["generation"] = mo.ui.dictionary(_patterns_gallery_generator_widget_map)

    for _patterns_corruption in pattern_corruptions[patterns_name_widget.value]:
        _patterns_corruption_name = _patterns_corruption.__name__
        _patterns_gallery_corruption_widget_map = {}
        _patterns_gallery_key_maps[_patterns_corruption_name] = {}
        for _patterns_parameter_name, _patterns_parameter_bounds in corruption_params[_patterns_corruption_name].items():
            _patterns_display_name = (
                _patterns_parameter_name[:-2] if _patterns_parameter_name.endswith(("_r", "_c")) else _patterns_parameter_name
            )
            _patterns_gallery_key_maps[_patterns_corruption_name][_patterns_display_name] = _patterns_parameter_name
            _patterns_parameter_low_value = float(_patterns_parameter_bounds[0])
            _patterns_parameter_high_value = float(_patterns_parameter_bounds[1])
            _patterns_gallery_corruption_widget_map[_patterns_display_name] = mo.ui.range_slider(
                start=_patterns_parameter_low_value,
                stop=_patterns_parameter_high_value,
                step=max((_patterns_parameter_high_value - _patterns_parameter_low_value) / 200.0, 0.0005),
                value=(_patterns_parameter_low_value, _patterns_parameter_high_value),
                show_value=True,
            )
        _patterns_gallery_sections[_patterns_corruption_name] = mo.ui.dictionary(_patterns_gallery_corruption_widget_map)

    patterns_gallery_key_maps = _patterns_gallery_key_maps
    patterns_gallery_ranges_widget = mo.ui.dictionary(_patterns_gallery_sections)
    return patterns_gallery_key_maps, patterns_gallery_ranges_widget


@app.cell
def _(full_ellipse, inspect, np, patterns_height_widget, patterns_single_generator_widget, patterns_width_widget):
    def _patterns_call_with_optional_rng(_patterns_function, *_patterns_args, _patterns_seed: int, **_patterns_kwargs):
        if "rng" in inspect.signature(_patterns_function).parameters:
            _patterns_kwargs["rng"] = np.random.default_rng(_patterns_seed)
        return _patterns_function(*_patterns_args, **_patterns_kwargs)

    patterns_single_shape = (int(patterns_height_widget.value), int(patterns_width_widget.value))
    patterns_single_stage_mask = full_ellipse(patterns_single_shape)
    patterns_single_generator_values = patterns_single_generator_widget.value
    patterns_single_call_with_optional_rng = _patterns_call_with_optional_rng
    return (
        patterns_single_call_with_optional_rng,
        patterns_single_generator_values,
        patterns_single_shape,
        patterns_single_stage_mask,
    )


@app.cell
def _(
    build_pattern_generator_kwargs,
    compact_heatmap,
    mo,
    np,
    pattern_generators,
    patterns_name_widget,
    patterns_seed_widget,
    patterns_single_generator_key_map,
    patterns_single_generator_values,
    patterns_single_generator_widget,
    patterns_single_shape,
    patterns_single_stage_mask,
    plt,
    patterns_single_call_with_optional_rng,
):
    _patterns_generator_function = pattern_generators[patterns_name_widget.value]
    _patterns_generator_kwargs = build_pattern_generator_kwargs(
        patterns_name_widget.value,
        patterns_single_generator_values,
        shape=patterns_single_shape,
        skip_parameters={"value_r", "polarity_c"},
    )

    patterns_single_generator_output = np.asarray(
        patterns_single_call_with_optional_rng(
            _patterns_generator_function,
            np.zeros(patterns_single_shape, dtype=np.float32),
            _patterns_seed=int(patterns_seed_widget.value) + 1,
            **_patterns_generator_kwargs,
        ),
        dtype=np.float32,
    )

    _patterns_generation_figure, _patterns_generation_axis = plt.subplots(figsize=(3.0, 3.0))
    compact_heatmap(
        np.where(patterns_single_stage_mask == 0.0, np.nan, patterns_single_generator_output),
        ax=_patterns_generation_axis,
        title="Wafer Map",
    )
    _patterns_generation_figure.tight_layout()
    _patterns_generation_widget = mo.ui.matplotlib(_patterns_generation_axis)
    plt.close(_patterns_generation_figure)
    mo.vstack(
        [
            mo.md("## 🎲 Generation"),
            mo.md(f"### {patterns_name_widget.value.replace('_', ' ').title()}"),
            mo.hstack(
                [patterns_single_generator_widget, _patterns_generation_widget],
                gap=2,
                widths=[1, 1],
                align="start",
            ),
        ],
        gap=1,
    )
    return (patterns_single_generator_output,)


@app.cell
def _(
    build_pattern_corruption_kwargs,
    compact_heatmap,
    mo,
    np,
    pattern_corruptions,
    patterns_name_widget,
    patterns_seed_widget,
    patterns_single_corruption_stage_1_controls,
    patterns_single_corruption_stage_1_key_map,
    patterns_single_corruption_stage_1_name,
    patterns_single_generator_output,
    patterns_single_stage_mask,
    plt,
    patterns_single_call_with_optional_rng,
):
    _patterns_corruption_functions = pattern_corruptions[patterns_name_widget.value]
    if len(_patterns_corruption_functions) == 0:
        patterns_single_corruption_stage_1_output = np.array(patterns_single_generator_output, copy=True)
        _patterns_corruption_view = mo.md("")
    else:
        _patterns_corruption_function = _patterns_corruption_functions[0]
        _patterns_corruption_name = patterns_single_corruption_stage_1_name
        _patterns_corruption_kwargs = {}
        for _patterns_display_name, _patterns_parameter_name in patterns_single_corruption_stage_1_key_map.items():
            _patterns_widget_value = patterns_single_corruption_stage_1_controls.value[_patterns_display_name]
            if _patterns_parameter_name.endswith("_r"):
                _patterns_corruption_kwargs[_patterns_parameter_name[:-2]] = float(_patterns_widget_value)
            elif _patterns_parameter_name.endswith("_c"):
                _patterns_corruption_kwargs[_patterns_parameter_name[:-2]] = _patterns_widget_value
        _patterns_corruption_kwargs = build_pattern_corruption_kwargs(
            _patterns_corruption_function,
            _patterns_corruption_kwargs,
            shape=patterns_single_generator_output.shape,
        )

        patterns_single_corruption_stage_1_output = np.asarray(
            patterns_single_call_with_optional_rng(
                _patterns_corruption_function,
                np.array(patterns_single_generator_output, copy=True),
                _patterns_seed=int(patterns_seed_widget.value) + 100,
                **_patterns_corruption_kwargs,
            ),
            dtype=np.float32,
        )

        _patterns_corruption_figure, _patterns_corruption_axis = plt.subplots(figsize=(3.0, 3.0))
        compact_heatmap(
            np.where(patterns_single_stage_mask == 0.0, np.nan, patterns_single_corruption_stage_1_output),
            ax=_patterns_corruption_axis,
            title="Wafer Map",
        )
        _patterns_corruption_figure.tight_layout()
        _patterns_corruption_widget = mo.ui.matplotlib(_patterns_corruption_axis)
        plt.close(_patterns_corruption_figure)
        _patterns_corruption_view = mo.vstack(
            [
                mo.md("## 🔨 Corruptions"),
                mo.md(f"### {_patterns_corruption_name.replace('_', ' ').title()}"),
                mo.hstack(
                    [patterns_single_corruption_stage_1_controls, _patterns_corruption_widget],
                    gap=2,
                    widths=[1, 1],
                    align="start",
                ),
            ],
            gap=1,
        )
    _patterns_corruption_view
    return (patterns_single_corruption_stage_1_output,)


@app.cell
def _(
    build_pattern_corruption_kwargs,
    compact_heatmap,
    mo,
    np,
    pattern_corruptions,
    patterns_name_widget,
    patterns_seed_widget,
    patterns_single_corruption_stage_1_output,
    patterns_single_corruption_stage_2_controls,
    patterns_single_corruption_stage_2_key_map,
    patterns_single_corruption_stage_2_name,
    patterns_single_stage_mask,
    plt,
    patterns_single_call_with_optional_rng,
):
    _patterns_corruption_functions = pattern_corruptions[patterns_name_widget.value]
    if len(_patterns_corruption_functions) <= 1:
        patterns_single_corruption_stage_2_output = np.array(patterns_single_corruption_stage_1_output, copy=True)
        _patterns_corruption_view = mo.md("")
    else:
        _patterns_corruption_function = _patterns_corruption_functions[1]
        _patterns_corruption_name = patterns_single_corruption_stage_2_name
        _patterns_corruption_kwargs = {}
        for _patterns_display_name, _patterns_parameter_name in patterns_single_corruption_stage_2_key_map.items():
            _patterns_widget_value = patterns_single_corruption_stage_2_controls.value[_patterns_display_name]
            if _patterns_parameter_name.endswith("_r"):
                _patterns_corruption_kwargs[_patterns_parameter_name[:-2]] = float(_patterns_widget_value)
            elif _patterns_parameter_name.endswith("_c"):
                _patterns_corruption_kwargs[_patterns_parameter_name[:-2]] = _patterns_widget_value
        _patterns_corruption_kwargs = build_pattern_corruption_kwargs(
            _patterns_corruption_function,
            _patterns_corruption_kwargs,
            shape=patterns_single_corruption_stage_1_output.shape,
        )

        patterns_single_corruption_stage_2_output = np.asarray(
            patterns_single_call_with_optional_rng(
                _patterns_corruption_function,
                np.array(patterns_single_corruption_stage_1_output, copy=True),
                _patterns_seed=int(patterns_seed_widget.value) + 101,
                **_patterns_corruption_kwargs,
            ),
            dtype=np.float32,
        )

        _patterns_corruption_figure, _patterns_corruption_axis = plt.subplots(figsize=(3.0, 3.0))
        compact_heatmap(
            np.where(patterns_single_stage_mask == 0.0, np.nan, patterns_single_corruption_stage_2_output),
            ax=_patterns_corruption_axis,
            title="Wafer Map",
        )
        _patterns_corruption_figure.tight_layout()
        _patterns_corruption_widget = mo.ui.matplotlib(_patterns_corruption_axis)
        plt.close(_patterns_corruption_figure)
        _patterns_corruption_view = mo.vstack(
            [
                mo.md(f"### {_patterns_corruption_name.replace('_', ' ').title()}"),
                mo.hstack(
                    [patterns_single_corruption_stage_2_controls, _patterns_corruption_widget],
                    gap=2,
                    widths=[1, 1],
                    align="start",
                ),
            ],
            gap=1,
        )
    _patterns_corruption_view
    return (patterns_single_corruption_stage_2_output,)


@app.cell
def _(
    build_pattern_corruption_kwargs,
    compact_heatmap,
    mo,
    np,
    pattern_corruptions,
    patterns_name_widget,
    patterns_seed_widget,
    patterns_single_corruption_stage_2_output,
    patterns_single_corruption_stage_3_controls,
    patterns_single_corruption_stage_3_key_map,
    patterns_single_corruption_stage_3_name,
    patterns_single_stage_mask,
    plt,
    patterns_single_call_with_optional_rng,
):
    _patterns_corruption_functions = pattern_corruptions[patterns_name_widget.value]
    if len(_patterns_corruption_functions) <= 2:
        patterns_single_corruption_stage_3_output = np.array(patterns_single_corruption_stage_2_output, copy=True)
        _patterns_corruption_view = mo.md("")
    else:
        _patterns_corruption_function = _patterns_corruption_functions[2]
        _patterns_corruption_name = patterns_single_corruption_stage_3_name
        _patterns_corruption_kwargs = {}
        for _patterns_display_name, _patterns_parameter_name in patterns_single_corruption_stage_3_key_map.items():
            _patterns_widget_value = patterns_single_corruption_stage_3_controls.value[_patterns_display_name]
            if _patterns_parameter_name.endswith("_r"):
                _patterns_corruption_kwargs[_patterns_parameter_name[:-2]] = float(_patterns_widget_value)
            elif _patterns_parameter_name.endswith("_c"):
                _patterns_corruption_kwargs[_patterns_parameter_name[:-2]] = _patterns_widget_value
        _patterns_corruption_kwargs = build_pattern_corruption_kwargs(
            _patterns_corruption_function,
            _patterns_corruption_kwargs,
            shape=patterns_single_corruption_stage_2_output.shape,
        )

        patterns_single_corruption_stage_3_output = np.asarray(
            patterns_single_call_with_optional_rng(
                _patterns_corruption_function,
                np.array(patterns_single_corruption_stage_2_output, copy=True),
                _patterns_seed=int(patterns_seed_widget.value) + 102,
                **_patterns_corruption_kwargs,
            ),
            dtype=np.float32,
        )

        _patterns_corruption_figure, _patterns_corruption_axis = plt.subplots(figsize=(3.0, 3.0))
        compact_heatmap(
            np.where(patterns_single_stage_mask == 0.0, np.nan, patterns_single_corruption_stage_3_output),
            ax=_patterns_corruption_axis,
            title="Wafer Map",
        )
        _patterns_corruption_figure.tight_layout()
        _patterns_corruption_widget = mo.ui.matplotlib(_patterns_corruption_axis)
        plt.close(_patterns_corruption_figure)
        _patterns_corruption_view = mo.vstack(
            [
                mo.md(f"### {_patterns_corruption_name.replace('_', ' ').title()}"),
                mo.hstack(
                    [patterns_single_corruption_stage_3_controls, _patterns_corruption_widget],
                    gap=2,
                    widths=[1, 1],
                    align="start",
                ),
            ],
            gap=1,
        )
    _patterns_corruption_view
    return (patterns_single_corruption_stage_3_output,)


@app.cell
def _(
    build_pattern_corruption_kwargs,
    compact_heatmap,
    mo,
    np,
    pattern_corruptions,
    patterns_name_widget,
    patterns_seed_widget,
    patterns_single_corruption_stage_3_output,
    patterns_single_corruption_stage_4_controls,
    patterns_single_corruption_stage_4_key_map,
    patterns_single_corruption_stage_4_name,
    patterns_single_stage_mask,
    plt,
    patterns_single_call_with_optional_rng,
):
    _patterns_corruption_functions = pattern_corruptions[patterns_name_widget.value]
    if len(_patterns_corruption_functions) <= 3:
        patterns_single_corruption_stage_4_output = np.array(patterns_single_corruption_stage_3_output, copy=True)
        _patterns_corruption_view = mo.md("")
    else:
        _patterns_corruption_function = _patterns_corruption_functions[3]
        _patterns_corruption_name = patterns_single_corruption_stage_4_name
        _patterns_corruption_kwargs = {}
        for _patterns_display_name, _patterns_parameter_name in patterns_single_corruption_stage_4_key_map.items():
            _patterns_widget_value = patterns_single_corruption_stage_4_controls.value[_patterns_display_name]
            if _patterns_parameter_name.endswith("_r"):
                _patterns_corruption_kwargs[_patterns_parameter_name[:-2]] = float(_patterns_widget_value)
            elif _patterns_parameter_name.endswith("_c"):
                _patterns_corruption_kwargs[_patterns_parameter_name[:-2]] = _patterns_widget_value
        _patterns_corruption_kwargs = build_pattern_corruption_kwargs(
            _patterns_corruption_function,
            _patterns_corruption_kwargs,
            shape=patterns_single_corruption_stage_3_output.shape,
        )

        patterns_single_corruption_stage_4_output = np.asarray(
            patterns_single_call_with_optional_rng(
                _patterns_corruption_function,
                np.array(patterns_single_corruption_stage_3_output, copy=True),
                _patterns_seed=int(patterns_seed_widget.value) + 103,
                **_patterns_corruption_kwargs,
            ),
            dtype=np.float32,
        )

        _patterns_corruption_figure, _patterns_corruption_axis = plt.subplots(figsize=(3.0, 3.0))
        compact_heatmap(
            np.where(patterns_single_stage_mask == 0.0, np.nan, patterns_single_corruption_stage_4_output),
            ax=_patterns_corruption_axis,
            title="Wafer Map",
        )
        _patterns_corruption_figure.tight_layout()
        _patterns_corruption_widget = mo.ui.matplotlib(_patterns_corruption_axis)
        plt.close(_patterns_corruption_figure)
        _patterns_corruption_view = mo.vstack(
            [
                mo.md(f"### {_patterns_corruption_name.replace('_', ' ').title()}"),
                mo.hstack(
                    [patterns_single_corruption_stage_4_controls, _patterns_corruption_widget],
                    gap=2,
                    widths=[1, 1],
                    align="start",
                ),
            ],
            gap=1,
        )
    _patterns_corruption_view
    return (patterns_single_corruption_stage_4_output,)


@app.cell
def _(
    build_pattern_corruption_kwargs,
    compact_heatmap,
    mo,
    np,
    pattern_corruptions,
    patterns_name_widget,
    patterns_seed_widget,
    patterns_single_corruption_stage_4_output,
    patterns_single_corruption_stage_5_controls,
    patterns_single_corruption_stage_5_key_map,
    patterns_single_corruption_stage_5_name,
    patterns_single_stage_mask,
    plt,
    patterns_single_call_with_optional_rng,
):
    _patterns_corruption_functions = pattern_corruptions[patterns_name_widget.value]
    if len(_patterns_corruption_functions) <= 4:
        patterns_single_corruption_stage_5_output = np.array(patterns_single_corruption_stage_4_output, copy=True)
        _patterns_corruption_view = mo.md("")
    else:
        _patterns_corruption_function = _patterns_corruption_functions[4]
        _patterns_corruption_name = patterns_single_corruption_stage_5_name
        _patterns_corruption_kwargs = {}
        for _patterns_display_name, _patterns_parameter_name in patterns_single_corruption_stage_5_key_map.items():
            _patterns_widget_value = patterns_single_corruption_stage_5_controls.value[_patterns_display_name]
            if _patterns_parameter_name.endswith("_r"):
                _patterns_corruption_kwargs[_patterns_parameter_name[:-2]] = float(_patterns_widget_value)
            elif _patterns_parameter_name.endswith("_c"):
                _patterns_corruption_kwargs[_patterns_parameter_name[:-2]] = _patterns_widget_value
        _patterns_corruption_kwargs = build_pattern_corruption_kwargs(
            _patterns_corruption_function,
            _patterns_corruption_kwargs,
            shape=patterns_single_corruption_stage_4_output.shape,
        )

        patterns_single_corruption_stage_5_output = np.asarray(
            patterns_single_call_with_optional_rng(
                _patterns_corruption_function,
                np.array(patterns_single_corruption_stage_4_output, copy=True),
                _patterns_seed=int(patterns_seed_widget.value) + 104,
                **_patterns_corruption_kwargs,
            ),
            dtype=np.float32,
        )

        _patterns_corruption_figure, _patterns_corruption_axis = plt.subplots(figsize=(3.0, 3.0))
        compact_heatmap(
            np.where(patterns_single_stage_mask == 0.0, np.nan, patterns_single_corruption_stage_5_output),
            ax=_patterns_corruption_axis,
            title="Wafer Map",
        )
        _patterns_corruption_figure.tight_layout()
        _patterns_corruption_widget = mo.ui.matplotlib(_patterns_corruption_axis)
        plt.close(_patterns_corruption_figure)
        _patterns_corruption_view = mo.vstack(
            [
                mo.md(f"### {_patterns_corruption_name.replace('_', ' ').title()}"),
                mo.hstack(
                    [patterns_single_corruption_stage_5_controls, _patterns_corruption_widget],
                    gap=2,
                    widths=[1, 1],
                    align="start",
                ),
            ],
            gap=1,
        )
    _patterns_corruption_view
    return (patterns_single_corruption_stage_5_output,)


@app.cell
def _(
    add_to_mask,
    add_to_pattern,
    combine_signals,
    init_output_dictionary,
    mask_sample,
    np,
    pattern2idx,
    patterns_name_widget,
    patterns_single_corruption_stage_5_output,
    patterns_single_generator_values,
    patterns_single_shape,
):
    patterns_single_stage_sample = init_output_dictionary(None, size=patterns_single_shape, num_classes=len(pattern2idx))
    patterns_single_stage_sample["temp"] = np.array(patterns_single_corruption_stage_5_output, copy=True)
    patterns_single_stage_sample = add_to_mask(patterns_single_stage_sample, idx=pattern2idx[patterns_name_widget.value])
    patterns_single_stage_sample = add_to_pattern(
        patterns_single_stage_sample,
        polarity=patterns_single_generator_values["polarity"],
        value=float(patterns_single_generator_values["value"]),
    )
    patterns_single_stage_sample = combine_signals(patterns_single_stage_sample)
    patterns_single_stage_sample = mask_sample(patterns_single_stage_sample)
    return (patterns_single_stage_sample,)


@app.cell
def _(
    compact_heatmap,
    mo,
    np,
    patterns_single_stage_mask,
    patterns_single_stage_sample,
    plot_compact_sinogram,
    plot_ecdf,
    plot_fft,
    plot_histogram,
    plt,
):
    _patterns_analysis_signal = np.where(
        patterns_single_stage_mask == 0.0,
        np.nan,
        np.asarray(patterns_single_stage_sample["sample"], dtype=np.float32),
    )

    _patterns_analysis_figure, _patterns_analysis_axes = plt.subplots(1, 5, figsize=(16, 3), layout="constrained")
    compact_heatmap(_patterns_analysis_signal, ax=_patterns_analysis_axes[0], title="Wafer Map")
    plot_fft(_patterns_analysis_signal, ax=_patterns_analysis_axes[1], create_fig=False)
    plot_compact_sinogram(_patterns_analysis_signal, ax=_patterns_analysis_axes[2], create_fig=False)
    plot_ecdf(_patterns_analysis_signal, ax=_patterns_analysis_axes[3], title="ECDF", create_fig=False)
    plot_histogram(_patterns_analysis_signal, ax=_patterns_analysis_axes[4], title="Histogram", create_fig=False)
    _patterns_analysis_widget = mo.ui.matplotlib(_patterns_analysis_axes[0])
    plt.close(_patterns_analysis_figure)

    _patterns_analysis_widget
    return


@app.cell
def _(
    add_to_mask,
    add_to_pattern,
    build_pattern_corruption_kwargs,
    build_pattern_generator_kwargs,
    combine_signals,
    compact_heatmap,
    full_ellipse,
    init_output_dictionary,
    inspect,
    mask_sample,
    mo,
    np,
    pattern2idx,
    pattern_corruptions,
    pattern_generators,
    patterns_gallery_key_maps,
    patterns_gallery_ranges_widget,
    patterns_height_widget,
    patterns_name_widget,
    patterns_seed_widget,
    patterns_width_widget,
    plot_mask,
    plt,
):
    _patterns_gallery_shape = (int(patterns_height_widget.value), int(patterns_width_widget.value))
    _patterns_gallery_mask = full_ellipse(_patterns_gallery_shape)
    _patterns_gallery_range_values = patterns_gallery_ranges_widget.value
    _patterns_signal_widgets = []
    _patterns_mask_widgets = []

    for _patterns_gallery_index in range(12):
        _patterns_gallery_sample = init_output_dictionary(None, size=_patterns_gallery_shape, num_classes=len(pattern2idx))
        _patterns_gallery_sample["temp"] = np.zeros(_patterns_gallery_shape, dtype=np.float32)

        _patterns_gallery_generator_function = pattern_generators[patterns_name_widget.value]
        _patterns_gallery_generator_signature = inspect.signature(_patterns_gallery_generator_function)
        _patterns_gallery_generator_rng = np.random.default_rng(int(patterns_seed_widget.value) + _patterns_gallery_index)
        _patterns_gallery_generator_values = {}
        for _patterns_display_name, _patterns_widget_value in _patterns_gallery_range_values["generation"].items():
            _patterns_parameter_name = patterns_gallery_key_maps["generation"][_patterns_display_name]
            if _patterns_parameter_name in {"value_r", "polarity_c"}:
                continue
            if _patterns_parameter_name.endswith("_r"):
                _patterns_gallery_generator_values[_patterns_display_name] = float(
                    _patterns_gallery_generator_rng.uniform(float(_patterns_widget_value[0]), float(_patterns_widget_value[1]))
                )
            elif _patterns_parameter_name.endswith("_c"):
                _patterns_gallery_generator_values[_patterns_display_name] = _patterns_widget_value
        _patterns_gallery_generator_kwargs = build_pattern_generator_kwargs(
            patterns_name_widget.value,
            _patterns_gallery_generator_values,
            shape=_patterns_gallery_shape,
            skip_parameters={"value_r", "polarity_c"},
        )
        if "rng" in _patterns_gallery_generator_signature.parameters:
            _patterns_gallery_generator_kwargs["rng"] = np.random.default_rng(
                int(patterns_seed_widget.value) + 1000 + _patterns_gallery_index
            )
        _patterns_gallery_sample["temp"] = np.asarray(
            _patterns_gallery_generator_function(_patterns_gallery_sample["temp"], **_patterns_gallery_generator_kwargs),
            dtype=np.float32,
        )

        for _patterns_corruption_index, _patterns_corruption_function in enumerate(pattern_corruptions[patterns_name_widget.value]):
            _patterns_corruption_name = _patterns_corruption_function.__name__
            _patterns_corruption_signature = inspect.signature(_patterns_corruption_function)
            _patterns_corruption_rng = np.random.default_rng(
                int(patterns_seed_widget.value) + 2000 + (_patterns_gallery_index * 10) + _patterns_corruption_index
            )
            _patterns_corruption_kwargs = {}
            for _patterns_display_name, _patterns_widget_value in _patterns_gallery_range_values[_patterns_corruption_name].items():
                _patterns_parameter_name = patterns_gallery_key_maps[_patterns_corruption_name][_patterns_display_name]
                if _patterns_parameter_name.endswith("_r"):
                    _patterns_corruption_kwargs[_patterns_parameter_name[:-2]] = float(
                        _patterns_corruption_rng.uniform(float(_patterns_widget_value[0]), float(_patterns_widget_value[1]))
                    )
                elif _patterns_parameter_name.endswith("_c"):
                    _patterns_corruption_kwargs[_patterns_parameter_name[:-2]] = _patterns_widget_value
            _patterns_corruption_kwargs = build_pattern_corruption_kwargs(
                _patterns_corruption_function,
                _patterns_corruption_kwargs,
                shape=_patterns_gallery_shape,
            )
            if "rng" in _patterns_corruption_signature.parameters:
                _patterns_corruption_kwargs["rng"] = np.random.default_rng(
                    int(patterns_seed_widget.value) + 3000 + (_patterns_gallery_index * 10) + _patterns_corruption_index
                )
            _patterns_gallery_sample["temp"] = np.asarray(
                _patterns_corruption_function(np.array(_patterns_gallery_sample["temp"], copy=True), **_patterns_corruption_kwargs),
                dtype=np.float32,
            )

        _patterns_gallery_sample = add_to_mask(_patterns_gallery_sample, idx=pattern2idx[patterns_name_widget.value])
        _patterns_gallery_sample = add_to_pattern(
            _patterns_gallery_sample,
            polarity=_patterns_gallery_range_values["generation"]["polarity"],
            value=float(
                np.random.default_rng(int(patterns_seed_widget.value) + 4000 + _patterns_gallery_index).uniform(
                    float(_patterns_gallery_range_values["generation"]["value"][0]),
                    float(_patterns_gallery_range_values["generation"]["value"][1]),
                )
            ),
        )
        _patterns_gallery_sample = combine_signals(_patterns_gallery_sample)
        _patterns_gallery_sample = mask_sample(_patterns_gallery_sample)

        _patterns_signal_array = np.where(
            _patterns_gallery_mask == 0.0,
            np.nan,
            np.asarray(_patterns_gallery_sample["sample"], dtype=np.float32),
        )
        _patterns_mask_array = _patterns_gallery_sample["masks"]

        _patterns_signal_figure, _patterns_signal_axis = plt.subplots(figsize=(2.4, 2.4))
        compact_heatmap(_patterns_signal_array, ax=_patterns_signal_axis)
        _patterns_signal_figure.tight_layout()
        _patterns_signal_widgets.append(mo.ui.matplotlib(_patterns_signal_axis))
        plt.close(_patterns_signal_figure)

        _patterns_mask_figure, _patterns_mask_axis = plt.subplots(figsize=(2.4, 2.4))
        plot_mask(_patterns_mask_array, create_fig=False, ax=_patterns_mask_axis, legend=False)
        _patterns_mask_figure.tight_layout()
        _patterns_mask_widgets.append(mo.ui.matplotlib(_patterns_mask_axis))
        plt.close(_patterns_mask_figure)

    _patterns_signal_rows = [
        mo.hstack(_patterns_signal_widgets[_patterns_row_index : _patterns_row_index + 4], justify="start", gap=1)
        for _patterns_row_index in range(0, len(_patterns_signal_widgets), 4)
    ]
    _patterns_mask_rows = [
        mo.hstack(_patterns_mask_widgets[_patterns_row_index : _patterns_row_index + 4], justify="start", gap=1)
        for _patterns_row_index in range(0, len(_patterns_mask_widgets), 4)
    ]
    _patterns_gallery_tabs = mo.ui.tabs(
        {
            "Signal": mo.vstack(_patterns_signal_rows, gap=1),
            "Masks": mo.vstack(_patterns_mask_rows, gap=1),
        },
        value="Signal",
    )

    mo.vstack(
        [
            mo.md("## 🖼️ Gallery"),
            patterns_gallery_ranges_widget,
            _patterns_gallery_tabs,
        ],
        gap=1,
    )
    return


if __name__ == "__main__":
    app.run()
