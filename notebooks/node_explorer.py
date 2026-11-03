# ruff: noqa: ANN, B018, E402, INP001, I001

"""🧩 Marimo node explorer."""

import marimo

__generated_with = "0.22.5"
app = marimo.App(width="full", app_title="➡️ Node Explorer")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from endlesswm.explorer_support import build_reference_inputs, fill_masked_array_with_nearest_valid_values
    from endlesswm.geometry import full_ellipse, linear_gradient, linear_radial_gradient
    from endlesswm.noise import beta_weibull, impulse_noise, pattern_noise, pink_noise
    from endlesswm.patterns import classes
    from endlesswm.transforms import gaussian_filter, local_pixel_shuffle, slant, smooth_normalize, swirl
    from endlesswm.trees import background_params, corruption_params
    from endlesswm.utils import affine
    from endlesswm.visualizations import compact_heatmap, plot_ecdf, plot_histogram

    return (
        affine,
        background_params,
        beta_weibull,
        build_reference_inputs,
        classes,
        compact_heatmap,
        corruption_params,
        fill_masked_array_with_nearest_valid_values,
        full_ellipse,
        gaussian_filter,
        impulse_noise,
        linear_gradient,
        linear_radial_gradient,
        local_pixel_shuffle,
        mo,
        np,
        pattern_noise,
        pink_noise,
        plot_ecdf,
        plot_histogram,
        plt,
        slant,
        smooth_normalize,
        swirl,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # ➡️ EndlessWM - Node Explorer
    """)
    return


@app.cell
def _(classes, mo):
    preview_height_widget = mo.ui.number(start=16, stop=128, step=1, value=52, label="height")
    preview_width_widget = mo.ui.number(start=16, stop=128, step=1, value=52, label="width")
    input_source_widget = mo.ui.dropdown(
        {
            "gaussian noise": "gaussian noise",
            "background map": "background map",
            "pattern signal": "pattern signal",
            "full wafer map sample": "full wafer map sample",
        },
        value="background map",
        label="transform input",
    )
    input_seed_widget = mo.ui.number(start=0, stop=100000, step=1, value=42, label="input seed")
    reference_pattern_widget = mo.ui.dropdown(
        {class_name: class_name for class_name in classes},
        value="center_spot",
        label="reference pattern",
    )
    top_section = mo.vstack(
        [
            mo.md("## 📐 Wafer Map Size"),
            mo.hstack([preview_height_widget, preview_width_widget], justify="start", gap=1),
            mo.md("## Generators"),
        ],
        gap=1,
    )
    top_section
    return (
        input_seed_widget,
        input_source_widget,
        preview_height_widget,
        preview_width_widget,
        reference_pattern_widget,
    )


@app.cell
def _(
    build_reference_inputs,
    full_ellipse,
    input_seed_widget,
    preview_height_widget,
    preview_width_widget,
    reference_pattern_widget,
):
    preview_shape = (int(preview_height_widget.value), int(preview_width_widget.value))
    wafer_mask = full_ellipse(preview_shape)
    reference_inputs = build_reference_inputs(
        seed=int(input_seed_widget.value),
        shape=preview_shape,
        reference_pattern=reference_pattern_widget.value,
    )
    reference_background_input = reference_inputs["background map"]
    reference_gaussian_input = reference_inputs["gaussian noise"]
    reference_pattern_input = reference_inputs["pattern signal"]
    reference_sample_input = reference_inputs["full wafer map sample"]
    return (
        preview_shape,
        reference_background_input,
        reference_gaussian_input,
        reference_pattern_input,
        reference_sample_input,
        wafer_mask,
    )


@app.cell
def _(background_params, mo):
    beta_weibull_seed_widget = mo.ui.number(start=0, stop=100000, step=1, value=42, label="seed")
    beta_weibull_parameter_tree = mo.ui.dictionary(
        {
            "alpha": mo.ui.slider(
                start=background_params["beta_weibull"]["alpha_r"][0],
                stop=background_params["beta_weibull"]["alpha_r"][1],
                step=0.01,
                value=1.0,
                show_value=True,
                include_input=True,
            ),
            "beta": mo.ui.slider(
                start=background_params["beta_weibull"]["beta_r"][0],
                stop=background_params["beta_weibull"]["beta_r"][1],
                step=0.01,
                value=1.0,
                show_value=True,
                include_input=True,
            ),
            "weight beta": mo.ui.slider(
                start=background_params["beta_weibull"]["weight_beta_r"][0],
                stop=background_params["beta_weibull"]["weight_beta_r"][1],
                step=0.01,
                value=0.5,
                show_value=True,
                include_input=True,
            ),
            "shape weibull": mo.ui.slider(
                start=background_params["beta_weibull"]["shape_weibull_r"][0],
                stop=background_params["beta_weibull"]["shape_weibull_r"][1],
                step=0.01,
                value=2.5,
                show_value=True,
                include_input=True,
            ),
            "weight weibull": mo.ui.slider(
                start=background_params["beta_weibull"]["weight_weibull_r"][0],
                stop=background_params["beta_weibull"]["weight_weibull_r"][1],
                step=0.01,
                value=0.5,
                show_value=True,
                include_input=True,
            ),
        },
        label="parameters",
    )
    beta_weibull_controls = mo.vstack(
        [mo.md("### 🔔 Beta Weibull"), mo.hstack([beta_weibull_seed_widget], justify="start"), beta_weibull_parameter_tree], gap=1
    )
    beta_weibull_controls
    return beta_weibull_parameter_tree, beta_weibull_seed_widget


@app.cell
def _(
    beta_weibull,
    beta_weibull_parameter_tree,
    beta_weibull_seed_widget,
    compact_heatmap,
    mo,
    np,
    plot_ecdf,
    plot_histogram,
    plt,
    preview_shape,
    wafer_mask,
):
    beta_weibull_output = beta_weibull(
        size=preview_shape,
        alpha=float(beta_weibull_parameter_tree.value["alpha"]),
        beta=float(beta_weibull_parameter_tree.value["beta"]),
        weight_beta=float(beta_weibull_parameter_tree.value["weight beta"]),
        shape_weibull=float(beta_weibull_parameter_tree.value["shape weibull"]),
        weight_weibull=float(beta_weibull_parameter_tree.value["weight weibull"]),
        rng=int(beta_weibull_seed_widget.value),
    )
    beta_weibull_masked = np.where(wafer_mask == 0.0, np.nan, np.asarray(beta_weibull_output, dtype=np.float32))
    beta_weibull_heatmap_figure, beta_weibull_heatmap_axis = plt.subplots(figsize=(3.0, 3.0))
    beta_weibull_ecdf_figure, beta_weibull_ecdf_axis = plt.subplots(figsize=(3.0, 3.0))
    beta_weibull_histogram_figure, beta_weibull_histogram_axis = plt.subplots(figsize=(3.0, 3.0))
    compact_heatmap(beta_weibull_masked, ax=beta_weibull_heatmap_axis, title="Wafer Map")
    plot_ecdf(beta_weibull_masked, ax=beta_weibull_ecdf_axis, title="ECDF", create_fig=False)
    plot_histogram(beta_weibull_masked, ax=beta_weibull_histogram_axis, title="Histogram", create_fig=False)
    mo.hstack(
        [
            mo.ui.matplotlib(beta_weibull_heatmap_axis),
            mo.ui.matplotlib(beta_weibull_ecdf_axis),
            mo.ui.matplotlib(beta_weibull_histogram_axis),
        ],
        widths="equal",
        gap=1,
    )
    return


@app.cell
def _(background_params, mo):
    pink_noise_seed_widget = mo.ui.number(start=0, stop=100000, step=1, value=43, label="seed")
    pink_noise_parameter_tree = mo.ui.dictionary(
        {
            "roughness": mo.ui.slider(
                start=background_params["pink_noise"]["roughness_r"][0],
                stop=background_params["pink_noise"]["roughness_r"][1],
                step=0.01,
                value=1.0,
                show_value=True,
                include_input=True,
            ),
            "max value": mo.ui.slider(
                start=background_params["pink_noise"]["max_value_r"][0],
                stop=background_params["pink_noise"]["max_value_r"][1],
                step=0.01,
                value=1.0,
                show_value=True,
                include_input=True,
            ),
        },
        label="parameters",
    )
    pink_noise_controls = mo.vstack(
        [mo.md("### 🪷 Pink Noise"), mo.hstack([pink_noise_seed_widget], justify="start"), pink_noise_parameter_tree], gap=1
    )
    pink_noise_controls
    return pink_noise_parameter_tree, pink_noise_seed_widget


@app.cell
def _(
    compact_heatmap,
    mo,
    np,
    pink_noise,
    pink_noise_parameter_tree,
    pink_noise_seed_widget,
    plot_ecdf,
    plot_histogram,
    plt,
    preview_shape,
    wafer_mask,
):
    pink_noise_output = pink_noise(
        size=preview_shape,
        roughness=float(pink_noise_parameter_tree.value["roughness"]),
        max_value=float(pink_noise_parameter_tree.value["max value"]),
        rng=int(pink_noise_seed_widget.value),
    )
    pink_noise_masked = np.where(wafer_mask == 0.0, np.nan, np.asarray(pink_noise_output, dtype=np.float32))
    pink_noise_heatmap_figure, pink_noise_heatmap_axis = plt.subplots(figsize=(3.0, 3.0))
    pink_noise_ecdf_figure, pink_noise_ecdf_axis = plt.subplots(figsize=(3.0, 3.0))
    pink_noise_histogram_figure, pink_noise_histogram_axis = plt.subplots(figsize=(3.0, 3.0))
    compact_heatmap(pink_noise_masked, ax=pink_noise_heatmap_axis, title="Wafer Map")
    plot_ecdf(pink_noise_masked, ax=pink_noise_ecdf_axis, title="ECDF", create_fig=False)
    plot_histogram(pink_noise_masked, ax=pink_noise_histogram_axis, title="Histogram", create_fig=False)
    mo.hstack(
        [mo.ui.matplotlib(pink_noise_heatmap_axis), mo.ui.matplotlib(pink_noise_ecdf_axis), mo.ui.matplotlib(pink_noise_histogram_axis)],
        widths="equal",
        gap=1,
    )
    return


@app.cell
def _(background_params, mo):
    impulse_noise_seed_widget = mo.ui.number(start=0, stop=100000, step=1, value=44, label="seed")
    impulse_noise_parameter_tree = mo.ui.dictionary(
        {
            "fraction log10": mo.ui.slider(
                start=background_params["impulse_noise"]["fraction_log10_r"][0],
                stop=background_params["impulse_noise"]["fraction_log10_r"][1],
                step=0.01,
                value=-2.0,
                show_value=True,
                include_input=True,
            ),
            "low": mo.ui.slider(
                start=background_params["impulse_noise"]["low_r"][0],
                stop=background_params["impulse_noise"]["low_r"][1],
                step=0.01,
                value=-0.25,
                show_value=True,
                include_input=True,
            ),
            "high": mo.ui.slider(
                start=background_params["impulse_noise"]["high_r"][0],
                stop=background_params["impulse_noise"]["high_r"][1],
                step=0.01,
                value=0.5,
                show_value=True,
                include_input=True,
            ),
        },
        label="parameters",
    )
    impulse_noise_controls = mo.vstack(
        [mo.md("### 🔨 Impulse Noise"), mo.hstack([impulse_noise_seed_widget], justify="start"), impulse_noise_parameter_tree], gap=1
    )
    impulse_noise_controls
    return impulse_noise_parameter_tree, impulse_noise_seed_widget


@app.cell
def _(
    compact_heatmap,
    impulse_noise,
    impulse_noise_parameter_tree,
    impulse_noise_seed_widget,
    mo,
    np,
    plot_ecdf,
    plot_histogram,
    plt,
    preview_shape,
    wafer_mask,
):
    impulse_noise_output = impulse_noise(
        size=preview_shape,
        fraction_log10=float(impulse_noise_parameter_tree.value["fraction log10"]),
        low=float(impulse_noise_parameter_tree.value["low"]),
        high=float(impulse_noise_parameter_tree.value["high"]),
        rng=int(impulse_noise_seed_widget.value),
    )
    impulse_noise_masked = np.where(wafer_mask == 0.0, np.nan, np.asarray(impulse_noise_output, dtype=np.float32))
    impulse_noise_heatmap_figure, impulse_noise_heatmap_axis = plt.subplots(figsize=(3.0, 3.0))
    impulse_noise_ecdf_figure, impulse_noise_ecdf_axis = plt.subplots(figsize=(3.0, 3.0))
    impulse_noise_histogram_figure, impulse_noise_histogram_axis = plt.subplots(figsize=(3.0, 3.0))
    compact_heatmap(impulse_noise_masked, ax=impulse_noise_heatmap_axis, title="Wafer Map")
    plot_ecdf(impulse_noise_masked, ax=impulse_noise_ecdf_axis, title="ECDF", create_fig=False)
    plot_histogram(impulse_noise_masked, ax=impulse_noise_histogram_axis, title="Histogram", create_fig=False)
    mo.hstack(
        [
            mo.ui.matplotlib(impulse_noise_heatmap_axis),
            mo.ui.matplotlib(impulse_noise_ecdf_axis),
            mo.ui.matplotlib(impulse_noise_histogram_axis),
        ],
        widths="equal",
        gap=1,
    )
    return


@app.cell
def _(background_params, mo):
    linear_gradient_parameter_tree = mo.ui.dictionary(
        {
            "angle": mo.ui.slider(
                start=background_params["linear_gradient"]["angle_r"][0],
                stop=background_params["linear_gradient"]["angle_r"][1],
                step=0.01,
                value=0.0,
                show_value=True,
                include_input=True,
            ),
            "slant fraction": mo.ui.slider(
                start=background_params["linear_gradient"]["slant_fraction_r"][0],
                stop=background_params["linear_gradient"]["slant_fraction_r"][1],
                step=0.01,
                value=0.125,
                show_value=True,
                include_input=True,
            ),
        },
        label="parameters",
    )
    linear_gradient_controls = mo.vstack([mo.md("### 📏 Linear Gradient"), linear_gradient_parameter_tree], gap=1)
    linear_gradient_controls
    return (linear_gradient_parameter_tree,)


@app.cell
def _(
    compact_heatmap,
    linear_gradient,
    linear_gradient_parameter_tree,
    mo,
    np,
    plot_ecdf,
    plot_histogram,
    plt,
    preview_shape,
    wafer_mask,
):
    linear_gradient_output = linear_gradient(
        shape=preview_shape,
        angle=float(linear_gradient_parameter_tree.value["angle"]),
        slant_fraction=float(linear_gradient_parameter_tree.value["slant fraction"]),
    )
    linear_gradient_masked = np.where(wafer_mask == 0.0, np.nan, np.asarray(linear_gradient_output, dtype=np.float32))
    linear_gradient_heatmap_figure, linear_gradient_heatmap_axis = plt.subplots(figsize=(3.0, 3.0))
    linear_gradient_ecdf_figure, linear_gradient_ecdf_axis = plt.subplots(figsize=(3.0, 3.0))
    linear_gradient_histogram_figure, linear_gradient_histogram_axis = plt.subplots(figsize=(3.0, 3.0))
    compact_heatmap(linear_gradient_masked, ax=linear_gradient_heatmap_axis, title="Wafer Map")
    plot_ecdf(linear_gradient_masked, ax=linear_gradient_ecdf_axis, title="ECDF", create_fig=False)
    plot_histogram(linear_gradient_masked, ax=linear_gradient_histogram_axis, title="Histogram", create_fig=False)
    mo.hstack(
        [
            mo.ui.matplotlib(linear_gradient_heatmap_axis),
            mo.ui.matplotlib(linear_gradient_ecdf_axis),
            mo.ui.matplotlib(linear_gradient_histogram_axis),
        ],
        widths="equal",
        gap=1,
    )
    return


@app.cell
def _(background_params, mo):
    linear_radial_gradient_parameter_tree = mo.ui.dictionary(
        {
            "center value": mo.ui.slider(
                start=background_params["linear_radial_gradient"]["center_val_r"][0],
                stop=background_params["linear_radial_gradient"]["center_val_r"][1],
                step=0.01,
                value=0.75,
                show_value=True,
                include_input=True,
            ),
            "outer value": mo.ui.slider(
                start=background_params["linear_radial_gradient"]["outer_val_r"][0],
                stop=background_params["linear_radial_gradient"]["outer_val_r"][1],
                step=0.01,
                value=1.25,
                show_value=True,
                include_input=True,
            ),
        },
        label="parameters",
    )
    linear_radial_gradient_controls = mo.vstack([mo.md("### 🛞 Linear Radial Gradient"), linear_radial_gradient_parameter_tree], gap=1)
    linear_radial_gradient_controls
    return (linear_radial_gradient_parameter_tree,)


@app.cell
def _(
    compact_heatmap,
    linear_radial_gradient,
    linear_radial_gradient_parameter_tree,
    mo,
    np,
    plot_ecdf,
    plot_histogram,
    plt,
    preview_shape,
    wafer_mask,
):
    linear_radial_gradient_output = linear_radial_gradient(
        shape=preview_shape,
        center_val=float(linear_radial_gradient_parameter_tree.value["center value"]),
        outer_val=float(linear_radial_gradient_parameter_tree.value["outer value"]),
    )
    linear_radial_gradient_masked = np.where(wafer_mask == 0.0, np.nan, np.asarray(linear_radial_gradient_output, dtype=np.float32))
    linear_radial_gradient_heatmap_figure, linear_radial_gradient_heatmap_axis = plt.subplots(figsize=(3.0, 3.0))
    linear_radial_gradient_ecdf_figure, linear_radial_gradient_ecdf_axis = plt.subplots(figsize=(3.0, 3.0))
    linear_radial_gradient_histogram_figure, linear_radial_gradient_histogram_axis = plt.subplots(figsize=(3.0, 3.0))
    compact_heatmap(linear_radial_gradient_masked, ax=linear_radial_gradient_heatmap_axis, title="Wafer Map")
    plot_ecdf(linear_radial_gradient_masked, ax=linear_radial_gradient_ecdf_axis, title="ECDF", create_fig=False)
    plot_histogram(linear_radial_gradient_masked, ax=linear_radial_gradient_histogram_axis, title="Histogram", create_fig=False)
    mo.hstack(
        [
            mo.ui.matplotlib(linear_radial_gradient_heatmap_axis),
            mo.ui.matplotlib(linear_radial_gradient_ecdf_axis),
            mo.ui.matplotlib(linear_radial_gradient_histogram_axis),
        ],
        widths="equal",
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Transformations
    """)
    return


@app.cell
def _(
    compact_heatmap,
    fill_masked_array_with_nearest_valid_values,
    input_seed_widget,
    input_source_widget,
    mo,
    np,
    plot_ecdf,
    plot_histogram,
    plt,
    reference_background_input,
    reference_gaussian_input,
    reference_pattern_input,
    reference_pattern_widget,
    reference_sample_input,
    wafer_mask,
):
    if input_source_widget.value == "gaussian noise":
        selected_input_array = reference_gaussian_input
    elif input_source_widget.value == "background map":
        selected_input_array = reference_background_input
    elif input_source_widget.value == "pattern signal":
        selected_input_array = reference_pattern_input
    else:
        selected_input_array = reference_sample_input
    selected_input_masked = np.where(wafer_mask == 0.0, np.nan, np.asarray(selected_input_array, dtype=np.float32))
    selected_input_heatmap_figure, selected_input_heatmap_axis = plt.subplots(figsize=(3.0, 3.0))
    selected_input_ecdf_figure, selected_input_ecdf_axis = plt.subplots(figsize=(3.0, 3.0))
    selected_input_histogram_figure, selected_input_histogram_axis = plt.subplots(figsize=(3.0, 3.0))
    compact_heatmap(selected_input_masked, ax=selected_input_heatmap_axis, title="Wafer Map")
    plot_ecdf(selected_input_masked, ax=selected_input_ecdf_axis, title="ECDF", create_fig=False)
    plot_histogram(selected_input_masked, ax=selected_input_histogram_axis, title="Histogram", create_fig=False)
    mo.vstack(
        [
            mo.md("### 🌐 Input"),
            mo.hstack([input_source_widget, input_seed_widget, reference_pattern_widget], justify="start", gap=1, wrap=True),
            mo.hstack(
                [
                    mo.ui.matplotlib(selected_input_heatmap_axis),
                    mo.ui.matplotlib(selected_input_ecdf_axis),
                    mo.ui.matplotlib(selected_input_histogram_axis),
                ],
                widths="equal",
                gap=1,
            ),
        ],
        gap=1,
    )
    selected_transform_input_array = fill_masked_array_with_nearest_valid_values(selected_input_array, wafer_mask)
    return (selected_transform_input_array,)


@app.cell
def _(background_params, mo):
    gaussian_filter_parameter_tree = mo.ui.dictionary(
        {
            "sigma": mo.ui.slider(
                start=background_params["gaussian_filter"]["sigma_r"][0],
                stop=background_params["gaussian_filter"]["sigma_r"][1],
                step=0.01,
                value=1.0,
                show_value=True,
                include_input=True,
            ),
        },
        label="parameters",
    )
    gaussian_filter_controls = mo.vstack([mo.md("### ☁️ Gaussian Filter"), gaussian_filter_parameter_tree], gap=1)
    gaussian_filter_controls
    return (gaussian_filter_parameter_tree,)


@app.cell
def _(
    compact_heatmap,
    gaussian_filter,
    gaussian_filter_parameter_tree,
    mo,
    np,
    plot_ecdf,
    plot_histogram,
    plt,
    selected_transform_input_array,
    wafer_mask,
):
    gaussian_filter_output = gaussian_filter(
        np.asarray(selected_transform_input_array, dtype=np.float32), sigma=float(gaussian_filter_parameter_tree.value["sigma"])
    )
    gaussian_filter_masked = np.where(wafer_mask == 0.0, np.nan, np.asarray(gaussian_filter_output, dtype=np.float32))
    gaussian_filter_heatmap_figure, gaussian_filter_heatmap_axis = plt.subplots(figsize=(3.0, 3.0))
    gaussian_filter_ecdf_figure, gaussian_filter_ecdf_axis = plt.subplots(figsize=(3.0, 3.0))
    gaussian_filter_histogram_figure, gaussian_filter_histogram_axis = plt.subplots(figsize=(3.0, 3.0))
    compact_heatmap(gaussian_filter_masked, ax=gaussian_filter_heatmap_axis, title="Wafer Map")
    plot_ecdf(gaussian_filter_masked, ax=gaussian_filter_ecdf_axis, title="ECDF", create_fig=False)
    plot_histogram(gaussian_filter_masked, ax=gaussian_filter_histogram_axis, title="Histogram", create_fig=False)
    mo.hstack(
        [
            mo.ui.matplotlib(gaussian_filter_heatmap_axis),
            mo.ui.matplotlib(gaussian_filter_ecdf_axis),
            mo.ui.matplotlib(gaussian_filter_histogram_axis),
        ],
        widths="equal",
        gap=1,
    )
    return


@app.cell
def _(background_params, mo):
    affine_parameter_tree = mo.ui.dictionary(
        {
            "offset": mo.ui.slider(
                start=background_params["affine"]["offset_r"][0],
                stop=background_params["affine"]["offset_r"][1],
                step=0.01,
                value=0.0,
                show_value=True,
                include_input=True,
            ),
            "scaling": mo.ui.slider(
                start=background_params["affine"]["scaling_r"][0],
                stop=background_params["affine"]["scaling_r"][1],
                step=0.01,
                value=1.0,
                show_value=True,
                include_input=True,
            ),
        },
        label="parameters",
    )
    affine_controls = mo.vstack([mo.md("### 📐 Affine"), affine_parameter_tree], gap=1)
    affine_controls
    return (affine_parameter_tree,)


@app.cell
def _(
    affine,
    affine_parameter_tree,
    compact_heatmap,
    mo,
    np,
    plot_ecdf,
    plot_histogram,
    plt,
    selected_transform_input_array,
    wafer_mask,
):
    affine_output = affine(
        np.asarray(selected_transform_input_array, dtype=np.float32),
        offset=float(affine_parameter_tree.value["offset"]),
        scaling=float(affine_parameter_tree.value["scaling"]),
    )
    affine_masked = np.where(wafer_mask == 0.0, np.nan, np.asarray(affine_output, dtype=np.float32))
    affine_heatmap_figure, affine_heatmap_axis = plt.subplots(figsize=(3.0, 3.0))
    affine_ecdf_figure, affine_ecdf_axis = plt.subplots(figsize=(3.0, 3.0))
    affine_histogram_figure, affine_histogram_axis = plt.subplots(figsize=(3.0, 3.0))
    compact_heatmap(affine_masked, ax=affine_heatmap_axis, title="Wafer Map")
    plot_ecdf(affine_masked, ax=affine_ecdf_axis, title="ECDF", create_fig=False)
    plot_histogram(affine_masked, ax=affine_histogram_axis, title="Histogram", create_fig=False)
    mo.hstack(
        [mo.ui.matplotlib(affine_heatmap_axis), mo.ui.matplotlib(affine_ecdf_axis), mo.ui.matplotlib(affine_histogram_axis)],
        widths="equal",
        gap=1,
    )
    return


@app.cell
def _(corruption_params, mo):
    pattern_noise_seed_widget = mo.ui.number(start=0, stop=100000, step=1, value=45, label="seed")
    pattern_noise_parameter_tree = mo.ui.dictionary(
        {
            "strength": mo.ui.slider(
                start=corruption_params["pattern_noise"]["strength_r"][0],
                stop=corruption_params["pattern_noise"]["strength_r"][1],
                step=0.005,
                value=0.1,
                show_value=True,
                include_input=True,
            ),
        },
        label="parameters",
    )
    pattern_noise_controls = mo.vstack(
        [mo.md("### 🔊 Pattern Noise"), mo.hstack([pattern_noise_seed_widget], justify="start"), pattern_noise_parameter_tree], gap=1
    )
    pattern_noise_controls
    return pattern_noise_parameter_tree, pattern_noise_seed_widget


@app.cell
def _(
    compact_heatmap,
    mo,
    np,
    pattern_noise,
    pattern_noise_parameter_tree,
    pattern_noise_seed_widget,
    plot_ecdf,
    plot_histogram,
    plt,
    selected_transform_input_array,
    wafer_mask,
):
    pattern_noise_output = pattern_noise(
        np.asarray(selected_transform_input_array, dtype=np.float32),
        strength=float(pattern_noise_parameter_tree.value["strength"]),
        rng=int(pattern_noise_seed_widget.value),
    )
    pattern_noise_masked = np.where(wafer_mask == 0.0, np.nan, np.asarray(pattern_noise_output, dtype=np.float32))
    pattern_noise_heatmap_figure, pattern_noise_heatmap_axis = plt.subplots(figsize=(3.0, 3.0))
    pattern_noise_ecdf_figure, pattern_noise_ecdf_axis = plt.subplots(figsize=(3.0, 3.0))
    pattern_noise_histogram_figure, pattern_noise_histogram_axis = plt.subplots(figsize=(3.0, 3.0))
    compact_heatmap(pattern_noise_masked, ax=pattern_noise_heatmap_axis, title="Wafer Map")
    plot_ecdf(pattern_noise_masked, ax=pattern_noise_ecdf_axis, title="ECDF", create_fig=False)
    plot_histogram(pattern_noise_masked, ax=pattern_noise_histogram_axis, title="Histogram", create_fig=False)
    mo.hstack(
        [
            mo.ui.matplotlib(pattern_noise_heatmap_axis),
            mo.ui.matplotlib(pattern_noise_ecdf_axis),
            mo.ui.matplotlib(pattern_noise_histogram_axis),
        ],
        widths="equal",
        gap=1,
    )
    return


@app.cell
def _(corruption_params, mo):
    local_pixel_shuffle_seed_widget = mo.ui.number(start=0, stop=100000, step=1, value=46, label="seed")
    local_pixel_shuffle_parameter_tree = mo.ui.dictionary(
        {
            "fraction": mo.ui.slider(
                start=corruption_params["local_pixel_shuffle"]["fraction_r"][0],
                stop=corruption_params["local_pixel_shuffle"]["fraction_r"][1],
                step=0.005,
                value=0.1,
                show_value=True,
                include_input=True,
            ),
            "radius": mo.ui.slider(
                start=corruption_params["local_pixel_shuffle"]["radius_r"][0],
                stop=corruption_params["local_pixel_shuffle"]["radius_r"][1],
                step=0.001,
                value=0.04,
                show_value=True,
                include_input=True,
            ),
        },
        label="parameters",
    )
    local_pixel_shuffle_controls = mo.vstack(
        [
            mo.md("### 👾 Local Pixel Shuffle"),
            mo.hstack([local_pixel_shuffle_seed_widget], justify="start"),
            local_pixel_shuffle_parameter_tree,
        ],
        gap=1,
    )
    local_pixel_shuffle_controls
    return local_pixel_shuffle_parameter_tree, local_pixel_shuffle_seed_widget


@app.cell
def _(
    compact_heatmap,
    local_pixel_shuffle,
    local_pixel_shuffle_parameter_tree,
    local_pixel_shuffle_seed_widget,
    mo,
    np,
    plot_ecdf,
    plot_histogram,
    plt,
    selected_transform_input_array,
    wafer_mask,
):
    local_pixel_shuffle_output = local_pixel_shuffle(
        np.asarray(selected_transform_input_array, dtype=np.float32),
        fraction=float(local_pixel_shuffle_parameter_tree.value["fraction"]),
        radius=float(local_pixel_shuffle_parameter_tree.value["radius"]),
        rng=int(local_pixel_shuffle_seed_widget.value),
    )
    local_pixel_shuffle_masked = np.where(wafer_mask == 0.0, np.nan, np.asarray(local_pixel_shuffle_output, dtype=np.float32))
    local_pixel_shuffle_heatmap_figure, local_pixel_shuffle_heatmap_axis = plt.subplots(figsize=(3.0, 3.0))
    local_pixel_shuffle_ecdf_figure, local_pixel_shuffle_ecdf_axis = plt.subplots(figsize=(3.0, 3.0))
    local_pixel_shuffle_histogram_figure, local_pixel_shuffle_histogram_axis = plt.subplots(figsize=(3.0, 3.0))
    compact_heatmap(local_pixel_shuffle_masked, ax=local_pixel_shuffle_heatmap_axis, title="Wafer Map")
    plot_ecdf(local_pixel_shuffle_masked, ax=local_pixel_shuffle_ecdf_axis, title="ECDF", create_fig=False)
    plot_histogram(local_pixel_shuffle_masked, ax=local_pixel_shuffle_histogram_axis, title="Histogram", create_fig=False)
    mo.hstack(
        [
            mo.ui.matplotlib(local_pixel_shuffle_heatmap_axis),
            mo.ui.matplotlib(local_pixel_shuffle_ecdf_axis),
            mo.ui.matplotlib(local_pixel_shuffle_histogram_axis),
        ],
        widths="equal",
        gap=1,
    )
    return


@app.cell
def _(corruption_params, mo):
    swirl_parameter_tree = mo.ui.dictionary(
        {
            "strength": mo.ui.slider(
                start=corruption_params["swirl"]["strength_r"][0],
                stop=corruption_params["swirl"]["strength_r"][1],
                step=0.01,
                value=0.5,
                show_value=True,
                include_input=True,
            ),
        },
        label="parameters",
    )
    swirl_controls = mo.vstack([mo.md("### 🌀 Swirl"), swirl_parameter_tree], gap=1)
    swirl_controls
    return (swirl_parameter_tree,)


@app.cell
def _(
    compact_heatmap,
    mo,
    np,
    plot_ecdf,
    plot_histogram,
    plt,
    selected_transform_input_array,
    swirl,
    swirl_parameter_tree,
    wafer_mask,
):
    swirl_output = swirl(
        np.asarray(selected_transform_input_array, dtype=np.float32),
        strength=float(swirl_parameter_tree.value["strength"]),
    )
    swirl_masked = np.where(wafer_mask == 0.0, np.nan, np.asarray(swirl_output, dtype=np.float32))
    swirl_heatmap_figure, swirl_heatmap_axis = plt.subplots(figsize=(3.0, 3.0))
    swirl_ecdf_figure, swirl_ecdf_axis = plt.subplots(figsize=(3.0, 3.0))
    swirl_histogram_figure, swirl_histogram_axis = plt.subplots(figsize=(3.0, 3.0))
    compact_heatmap(swirl_masked, ax=swirl_heatmap_axis, title="Wafer Map")
    plot_ecdf(swirl_masked, ax=swirl_ecdf_axis, title="ECDF", create_fig=False)
    plot_histogram(swirl_masked, ax=swirl_histogram_axis, title="Histogram", create_fig=False)
    mo.hstack(
        [mo.ui.matplotlib(swirl_heatmap_axis), mo.ui.matplotlib(swirl_ecdf_axis), mo.ui.matplotlib(swirl_histogram_axis)],
        widths="equal",
        gap=1,
    )
    return


@app.cell
def _(corruption_params, mo):
    smooth_normalize_parameter_tree = mo.ui.dictionary(
        {
            "sigma": mo.ui.slider(
                start=corruption_params["smooth_normalize"]["sigma_r"][0],
                stop=corruption_params["smooth_normalize"]["sigma_r"][1],
                step=0.01,
                value=1.0,
                show_value=True,
                include_input=True,
            ),
        },
        label="parameters",
    )
    smooth_normalize_controls = mo.vstack([mo.md("### 🎨 Smooth Normalize"), smooth_normalize_parameter_tree], gap=1)
    smooth_normalize_controls
    return (smooth_normalize_parameter_tree,)


@app.cell
def _(
    compact_heatmap,
    mo,
    np,
    plot_ecdf,
    plot_histogram,
    plt,
    selected_transform_input_array,
    smooth_normalize,
    smooth_normalize_parameter_tree,
    wafer_mask,
):
    smooth_normalize_output = smooth_normalize(
        np.asarray(selected_transform_input_array, dtype=np.float32), sigma=float(smooth_normalize_parameter_tree.value["sigma"])
    )
    smooth_normalize_masked = np.where(wafer_mask == 0.0, np.nan, np.asarray(smooth_normalize_output, dtype=np.float32))
    smooth_normalize_heatmap_figure, smooth_normalize_heatmap_axis = plt.subplots(figsize=(3.0, 3.0))
    smooth_normalize_ecdf_figure, smooth_normalize_ecdf_axis = plt.subplots(figsize=(3.0, 3.0))
    smooth_normalize_histogram_figure, smooth_normalize_histogram_axis = plt.subplots(figsize=(3.0, 3.0))
    compact_heatmap(smooth_normalize_masked, ax=smooth_normalize_heatmap_axis, title="Wafer Map")
    plot_ecdf(smooth_normalize_masked, ax=smooth_normalize_ecdf_axis, title="ECDF", create_fig=False)
    plot_histogram(smooth_normalize_masked, ax=smooth_normalize_histogram_axis, title="Histogram", create_fig=False)
    mo.hstack(
        [
            mo.ui.matplotlib(smooth_normalize_heatmap_axis),
            mo.ui.matplotlib(smooth_normalize_ecdf_axis),
            mo.ui.matplotlib(smooth_normalize_histogram_axis),
        ],
        widths="equal",
        gap=1,
    )
    return


@app.cell
def _(corruption_params, mo):
    slant_parameter_tree = mo.ui.dictionary(
        {
            "scale": mo.ui.slider(
                start=corruption_params["slant"]["scale_r"][0],
                stop=corruption_params["slant"]["scale_r"][1],
                step=0.001,
                value=1.0,
                show_value=True,
                include_input=True,
            ),
            "rotation": mo.ui.slider(
                start=corruption_params["slant"]["rotation_r"][0],
                stop=corruption_params["slant"]["rotation_r"][1],
                step=0.001,
                value=0.0,
                show_value=True,
                include_input=True,
            ),
            "shear": mo.ui.slider(
                start=corruption_params["slant"]["shear_r"][0],
                stop=corruption_params["slant"]["shear_r"][1],
                step=0.001,
                value=0.0,
                show_value=True,
                include_input=True,
            ),
        },
        label="parameters",
    )
    slant_controls = mo.vstack([mo.md("### 🔄 Slant"), slant_parameter_tree], gap=1)
    slant_controls
    return (slant_parameter_tree,)


@app.cell
def _(
    compact_heatmap,
    mo,
    np,
    plot_ecdf,
    plot_histogram,
    plt,
    selected_transform_input_array,
    slant,
    slant_parameter_tree,
    wafer_mask,
):
    slant_output = slant(
        np.asarray(selected_transform_input_array, dtype=np.float32),
        scale=float(slant_parameter_tree.value["scale"]),
        rotation=float(slant_parameter_tree.value["rotation"]),
        shear=float(slant_parameter_tree.value["shear"]),
    )
    slant_masked = np.where(wafer_mask == 0.0, np.nan, np.asarray(slant_output, dtype=np.float32))
    slant_heatmap_figure, slant_heatmap_axis = plt.subplots(figsize=(3.0, 3.0))
    slant_ecdf_figure, slant_ecdf_axis = plt.subplots(figsize=(3.0, 3.0))
    slant_histogram_figure, slant_histogram_axis = plt.subplots(figsize=(3.0, 3.0))
    compact_heatmap(slant_masked, ax=slant_heatmap_axis, title="Wafer Map")
    plot_ecdf(slant_masked, ax=slant_ecdf_axis, title="ECDF", create_fig=False)
    plot_histogram(slant_masked, ax=slant_histogram_axis, title="Histogram", create_fig=False)
    mo.hstack(
        [mo.ui.matplotlib(slant_heatmap_axis), mo.ui.matplotlib(slant_ecdf_axis), mo.ui.matplotlib(slant_histogram_axis)],
        widths="equal",
        gap=1,
    )
    return


if __name__ == "__main__":
    app.run()
