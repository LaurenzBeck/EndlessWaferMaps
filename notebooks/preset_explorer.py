# ruff: noqa: ANN, B018, E402, INP001, I001

"""🌳 Marimo preset explorer."""

import marimo

__generated_with = "0.22.0"
app = marimo.App(width="full", app_title="🌳 Preset Explorer")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from endlesswm.cli.generation import build_tree_from_preset
    from endlesswm.explorer_support import (
        build_default_gif_path_from_dataset_path,
        build_explorer_gif_path,
        open_path_in_file_browser,
        render_sample_grid_frame,
        save_sample_grid_gif,
    )
    from endlesswm.io.export import filter_generation_parameters
    from endlesswm.patterns import classes as DEFAULT_CLASS_NAMES
    from endlesswm.pipeline.config.naming import resolve_output_path
    from endlesswm.pipeline.config.loader import list_all_drift_presets, load_drift_preset
    from endlesswm.visualizations import (
        build_parameter_evolution_heatmap,
        build_parameter_evolution_matrix,
        build_single_parameter_schedule_figure,
        compact_heatmap,
        plot_mask,
        resolve_scheduled_parameter_name,
        resolve_numeric_parameter_names,
    )

    return (
        DEFAULT_CLASS_NAMES,
        build_parameter_evolution_heatmap,
        build_parameter_evolution_matrix,
        build_default_gif_path_from_dataset_path,
        build_explorer_gif_path,
        build_single_parameter_schedule_figure,
        build_tree_from_preset,
        compact_heatmap,
        filter_generation_parameters,
        list_all_drift_presets,
        load_drift_preset,
        mo,
        np,
        open_path_in_file_browser,
        pd,
        plot_mask,
        plt,
        render_sample_grid_frame,
        resolve_output_path,
        resolve_scheduled_parameter_name,
        resolve_numeric_parameter_names,
        save_sample_grid_gif,
    )


@app.cell
def _(
    build_parameter_evolution_heatmap,
    build_parameter_evolution_matrix,
    build_single_parameter_schedule_figure,
    build_tree_from_preset,
    compact_heatmap,
    filter_generation_parameters,
    load_drift_preset,
    np,
    pd,
    plot_mask,
    plt,
    resolve_scheduled_parameter_name,
):
    def make_parameter_groups(parameter_names: list[str], class_names: list[str]) -> dict[str, list[str]]:
        groups = {"🌫️ background": []}
        for class_name in class_names:
            groups[f"🧩 {class_name.replace('_', ' ')}"] = []
        class_name_set = set(class_names)
        for parameter_name in parameter_names:
            root_name = parameter_name.split(".", 1)[0]
            group_name = f"🧩 {root_name.replace('_', ' ')}" if root_name in class_name_set else "🌫️ background"
            groups[group_name].append(parameter_name)
        return {group_name: names for group_name, names in groups.items() if names}

    def build_schedule_tables(preset_identifier: str):
        preset = load_drift_preset(preset_identifier, validate_runtime_capabilities=False)
        tree = build_tree_from_preset(preset)
        raw_df = tree.params.to_dataframe(number_of_update_steps=max(preset.num_tasks - 1, 0))
        parameter_rows = []

        for task_index in range(preset.num_tasks):
            sample_count = preset.num_samples_first_task if task_index == 0 else preset.num_samples_per_task
            if sample_count is None:
                sample_count = preset.num_samples_per_task
            sample_count = min(int(sample_count), 64)

            for sample_index in range(sample_count):
                sample_dict = tree.sample()
                flat_params = filter_generation_parameters(sample_dict["parameters"])
                parameter_rows.append(
                    {
                        "task": task_index,
                        "sample_idx": sample_index,
                        **flat_params,
                    },
                )

            if task_index < preset.num_tasks - 1:
                tree.update()

        parameter_df = pd.DataFrame(parameter_rows)
        return preset, raw_df, parameter_df

    def build_parameter_heatmap(parameter_df: pd.DataFrame, group_parameter_names: list[str]):
        matrix = build_parameter_evolution_matrix(
            parameter_df,
            parameter_names=group_parameter_names,
            statistic="median",
            include_constant_parameters=False,
            excluded_name_fragments=("polarity",),
        )
        if matrix.empty:
            matrix = build_parameter_evolution_matrix(
                parameter_df,
                parameter_names=group_parameter_names,
                statistic="median",
                include_constant_parameters=True,
                excluded_name_fragments=("polarity",),
            )
        return build_parameter_evolution_heatmap(
            matrix,
            title="",
            normalize_rows=True,
            normalize_with_full_allowed_range=True,
            show_parameter_labels=False,
        )

    def build_bounds(raw_df: pd.DataFrame, parameter_name: str) -> pd.DataFrame:
        schedule_parameter_name = resolve_scheduled_parameter_name(raw_df, parameter_name)
        if schedule_parameter_name is None:
            return pd.DataFrame({"task": np.arange(len(raw_df))})
        return pd.DataFrame(
            {
                "task": np.arange(len(raw_df)),
                "scheduled_min": [float(bounds[0]) for bounds in raw_df[f"{schedule_parameter_name}_r"]],
                "scheduled_max": [float(bounds[1]) for bounds in raw_df[f"{schedule_parameter_name}_r"]],
            },
        )

    def build_single_parameter_schedule_section(
        parameter_name: str,
        raw_df: pd.DataFrame,
    ):
        bounds_df = build_bounds(raw_df, parameter_name)
        return build_single_parameter_schedule_figure(bounds_df, parameter_name)

    def sample_task_arrays(preset_identifier: str, task_index: int, sample_view: str, num_samples: int = 36) -> list[np.ndarray]:
        preset = load_drift_preset(preset_identifier, validate_runtime_capabilities=False)
        tree = build_tree_from_preset(preset)
        for _ in range(task_index):
            tree.update()
        return [tree.sample()["masks" if sample_view == "mask annotations" else sample_view] for _ in range(num_samples)]

    def build_sample_gallery(mo, samples: list[np.ndarray], sample_view: str, columns: int = 6):
        figures = []
        for sample in samples:
            fig, ax = plt.subplots(figsize=(2.4, 2.4))
            if sample_view == "mask annotations":
                plot_mask(sample, create_fig=False, ax=ax, legend=False)
            else:
                if sample_view == "pattern":
                    sample = np.where(sample == 0, np.nan, sample)
                compact_heatmap(sample, ax=ax)
            fig.tight_layout()
            figures.append(fig)
            plt.close(fig)
        rows = [mo.hstack(figures[index : index + columns], justify="start") for index in range(0, len(figures), columns)]
        return mo.vstack(rows)

    return (
        build_parameter_heatmap,
        build_sample_gallery,
        build_schedule_tables,
        build_single_parameter_schedule_section,
        make_parameter_groups,
        sample_task_arrays,
    )


@app.cell
def _(list_all_drift_presets, mo):
    preset_names = list_all_drift_presets()
    preset_widget = mo.ui.dropdown({name: name for name in preset_names}, value=preset_names[0], label="🌳 preset")
    return (preset_widget,)


@app.cell
def _(preset_widget):
    selected_preset = preset_widget.value
    return (selected_preset,)


@app.cell
def _(build_schedule_tables, selected_preset):
    preset, raw_schedule_df, parameter_df = build_schedule_tables(selected_preset)
    return parameter_df, preset, raw_schedule_df


@app.cell
def _(
    DEFAULT_CLASS_NAMES,
    make_parameter_groups,
    parameter_df,
    resolve_numeric_parameter_names,
):
    parameter_names, _ = resolve_numeric_parameter_names(
        parameter_df,
        include_constant_parameters=True,
        excluded_name_fragments=("polarity",),
    )
    grouped_parameter_names = make_parameter_groups(parameter_names, DEFAULT_CLASS_NAMES)
    first_group = next(iter(grouped_parameter_names))
    return first_group, grouped_parameter_names


@app.cell
def _(first_group, grouped_parameter_names, mo):
    heatmap_group_widget = mo.ui.dropdown({name: name for name in grouped_parameter_names}, value=first_group, label="🧭 group")
    return (heatmap_group_widget,)


@app.cell
def _(first_group, grouped_parameter_names, mo):
    schedule_group_widget = mo.ui.dropdown({name: name for name in grouped_parameter_names}, value=first_group, label="🧭 group")
    return (schedule_group_widget,)


@app.cell
def _(grouped_parameter_names, mo, schedule_group_widget):
    selected_group_parameter_names = grouped_parameter_names[schedule_group_widget.value]
    schedule_parameter_widget = mo.ui.dropdown(
        {name: name for name in selected_group_parameter_names},
        value=selected_group_parameter_names[0],
        label="🎚️ param",
    )
    return (schedule_parameter_widget,)


@app.cell
def _(mo, preset):
    _max_task_index = max(0, preset.num_tasks - 1)
    task_slider_widget = mo.ui.slider(
        start=0,
        stop=_max_task_index,
        step=1,
        value=0,
        label="🎚️ task",
        include_input=True,
        show_value=True,
    )
    sample_view_widget = mo.ui.dropdown(
        {
            "sample": "sample",
            "pattern signal": "pattern",
            "background": "background",
            "mask annotations": "mask annotations",
        },
        value="sample",
        label="🖼️ view",
    )
    return sample_view_widget, task_slider_widget


@app.cell
def _(build_default_gif_path_from_dataset_path, mo, preset, resolve_output_path):
    _max_task_index = max(0, int(preset.num_tasks) - 1)
    _default_output_path = build_default_gif_path_from_dataset_path(resolve_output_path(preset)).as_posix()

    def _validate_preset_gif_form(values):
        if values is None:
            return None
        if int(values["end_task"]) < int(values["start_task"]):
            return "End task must be greater than or equal to start task."
        if int(values["task_stride"]) <= 0:
            return "Task stride must be greater than zero."
        if int(values["rows"]) * int(values["columns"]) <= 0:
            return "Choose at least one row and one column."
        if int(values["frame_duration_ms"]) <= 0:
            return "Frame length must be greater than zero."
        return None

    preset_gif_form = (
        mo.md(
            """
### 🎞️ GIF settings

{sample_view}
{start_task}
{end_task}
{task_stride}
{rows}
{columns}
{frame_duration_ms}
{output_path}
"""
        )
        .batch(
            sample_view=mo.ui.dropdown(
                {
                    "sample": "sample",
                    "pattern signal": "pattern",
                    "background": "background",
                    "mask annotations": "mask annotations",
                },
                value="sample",
                label="view",
            ),
            start_task=mo.ui.number(start=0, stop=_max_task_index, step=1, value=0, label="start task"),
            end_task=mo.ui.number(start=0, stop=_max_task_index, step=1, value=_max_task_index, label="end task"),
            task_stride=mo.ui.number(start=1, stop=max(_max_task_index + 1, 1), step=1, value=1, label="task stride"),
            rows=mo.ui.number(start=1, step=1, value=4, label="rows"),
            columns=mo.ui.number(start=1, step=1, value=8, label="columns"),
            frame_duration_ms=mo.ui.number(start=50, stop=5000, step=50, value=500, label="frame length (ms)"),
            output_path=mo.ui.text_area(value=_default_output_path, label="output path", full_width=True),
        )
        .form(
            submit_button_label="Create GIF",
            show_clear_button=True,
            clear_button_label="Reset",
            validate=_validate_preset_gif_form,
        )
    )
    return (preset_gif_form,)


@app.cell
def _(
    mo,
):
    get_preset_gif_state, set_preset_gif_state = mo.state({"request": None, "saved_path": None})
    return get_preset_gif_state, set_preset_gif_state


@app.cell
def _(
    build_parameter_heatmap,
    grouped_parameter_names,
    heatmap_group_widget,
    mo,
    parameter_df,
):
    heatmap_controls = mo.hstack([heatmap_group_widget], justify="start", wrap=True)
    heatmap_body = mo.vstack(
        [
            heatmap_controls,
            mo.lazy(
                lambda: build_parameter_heatmap(
                    parameter_df,
                    grouped_parameter_names[heatmap_group_widget.value],
                ),
                show_loading_indicator=True,
            ),
        ],
        gap=1,
    )
    heatmap_accordion = mo.accordion({"📊 parameter heatmaps": heatmap_body})
    return (heatmap_accordion,)


@app.cell
def _(
    build_single_parameter_schedule_section,
    mo,
    raw_schedule_df,
    schedule_group_widget,
    schedule_parameter_widget,
):
    schedule_controls = mo.hstack([schedule_group_widget, schedule_parameter_widget], justify="start", wrap=True)
    schedule_body = mo.vstack(
        [
            schedule_controls,
            mo.lazy(
                lambda: build_single_parameter_schedule_section(
                    schedule_parameter_widget.value,
                    raw_schedule_df,
                ),
                show_loading_indicator=True,
            ),
        ],
        gap=1,
    )
    single_parameter_accordion = mo.accordion({"🎚️ single parameter schedules": schedule_body})
    return (single_parameter_accordion,)


@app.cell
def _(
    build_sample_gallery,
    mo,
    sample_task_arrays,
    sample_view_widget,
    selected_preset,
    task_slider_widget,
):
    samples_accordion = None
    if sample_view_widget is not None and task_slider_widget is not None:
        sample_controls = mo.hstack([task_slider_widget, sample_view_widget], justify="start", wrap=True)
        sample_body = mo.vstack(
            [
                sample_controls,
                mo.lazy(
                    lambda: build_sample_gallery(
                        mo,
                        sample_task_arrays(selected_preset, int(task_slider_widget.value), sample_view_widget.value, 36),
                        sample_view_widget.value,
                        columns=6,
                    ),
                    show_loading_indicator=True,
                ),
            ],
            gap=1,
        )
        samples_accordion = mo.accordion({"🖼️ samples": sample_body})
    return (samples_accordion,)


@app.cell
def _(
    build_default_gif_path_from_dataset_path,
    build_explorer_gif_path,
    build_tree_from_preset,
    heatmap_accordion,
    get_preset_gif_state,
    load_drift_preset,
    mo,
    preset_gif_form,
    preset_widget,
    render_sample_grid_frame,
    resolve_output_path,
    save_sample_grid_gif,
    selected_preset,
    set_preset_gif_state,
    samples_accordion,
    single_parameter_accordion,
):
    def _build_preset_sections(preset_gif_accordion):
        sections = [preset_widget, heatmap_accordion, single_parameter_accordion, preset_gif_accordion]
        if samples_accordion is not None:
            sections.append(samples_accordion)
        return mo.vstack(sections, gap=1)

    def _build_progress_block(title: str, detail: str, current: int, total: int):
        safe_total = max(int(total), 1)
        safe_current = max(0, min(int(current), safe_total))
        return mo.vstack(
            [
                mo.md(f"**{title}**"),
                mo.Html(f'<progress value="{safe_current}" max="{safe_total}" style="width: 100%; height: 1rem;"></progress>'),
                mo.md(f"{detail} ({safe_current}/{safe_total})"),
            ],
            gap=0.5,
        )

    def _build_preset_gif_accordion(progress_block=None, result_block=None):
        if preset_gif_form is None:
            return None

        gif_sections = [preset_gif_form]
        if progress_block is not None:
            gif_sections.append(progress_block)
        if result_block is not None:
            gif_sections.append(result_block)
        return mo.accordion({"🎞️ animated sample GIF": mo.vstack(gif_sections, gap=1)})

    preset_gif_result_block = None
    preset_gif_progress_block = None
    preset_gif_state = get_preset_gif_state()
    preset_gif_request = None
    preset_gif_saved_path = None
    preset_open_gif_folder_button = None

    if preset_gif_form is not None and preset_gif_form.value is not None:
        _gif_settings = preset_gif_form.value
        _start_task = int(_gif_settings["start_task"])
        _end_task = int(_gif_settings["end_task"])
        _task_stride = int(_gif_settings["task_stride"])
        _rows = int(_gif_settings["rows"])
        _columns = int(_gif_settings["columns"])
        _sample_view = str(_gif_settings["sample_view"])
        _frame_duration_ms = int(_gif_settings["frame_duration_ms"])
        _samples_per_frame = _rows * _columns
        _gif_preset = load_drift_preset(selected_preset, validate_runtime_capabilities=False)
        _output_path = build_explorer_gif_path(
            str(_gif_settings["output_path"]),
            default_output_path=build_default_gif_path_from_dataset_path(resolve_output_path(_gif_preset)),
        )
        _task_indices = list(range(_start_task, _end_task + 1, _task_stride))
        preset_gif_request = {
            "preset": selected_preset,
            "sample_view": _sample_view,
            "start_task": _start_task,
            "end_task": _end_task,
            "task_stride": _task_stride,
            "rows": _rows,
            "columns": _columns,
            "frame_duration_ms": _frame_duration_ms,
            "output_path": _output_path.as_posix(),
        }

        if preset_gif_state.get("request") == preset_gif_request and preset_gif_state.get("saved_path"):
            preset_gif_saved_path = str(preset_gif_state["saved_path"])
        else:
            preset_gif_progress_block = _build_progress_block(
                "Creating GIF",
                "Preparing frame rendering",
                0,
                len(_task_indices),
            )
            mo.output.replace(_build_preset_sections(_build_preset_gif_accordion(progress_block=preset_gif_progress_block)))

            _tree = build_tree_from_preset(_gif_preset)
            for _ in range(_start_task):
                _tree.update()

            _frames = []
            _field_name = "masks" if _sample_view == "mask annotations" else _sample_view
            for frame_index, task_index in enumerate(_task_indices, start=1):
                preset_gif_progress_block = _build_progress_block(
                    "Rendering frames",
                    f"Rendering frame {frame_index} of {len(_task_indices)}",
                    frame_index - 1,
                    len(_task_indices),
                )
                mo.output.replace(_build_preset_sections(_build_preset_gif_accordion(progress_block=preset_gif_progress_block)))
                _samples = [_tree.sample()[_field_name] for _ in range(_samples_per_frame)]
                _frames.append(
                    render_sample_grid_frame(
                        _samples,
                        _sample_view,
                        _rows,
                        _columns,
                    )
                )
                if task_index < _end_task:
                    _tree.update()

            preset_gif_progress_block = _build_progress_block(
                "Saving GIF",
                f"Writing {_output_path.name}",
                len(_task_indices),
                len(_task_indices),
            )
            mo.output.replace(_build_preset_sections(_build_preset_gif_accordion(progress_block=preset_gif_progress_block)))
            preset_gif_saved_path = save_sample_grid_gif(_frames, _output_path, _frame_duration_ms).as_posix()
            set_preset_gif_state({"request": preset_gif_request, "saved_path": preset_gif_saved_path})

        preset_open_gif_folder_button = mo.ui.run_button(label="Open folder")

        preset_gif_result_block = mo.vstack(
            [
                mo.md(f"Saved GIF to `{preset_gif_saved_path}`"),
                preset_open_gif_folder_button,
                mo.md(
                    f"Created {len(_task_indices)} frame(s) with a {_rows}x{_columns} grid and {_samples_per_frame} sample(s) per frame."
                ),
                mo.image(src=preset_gif_saved_path),
            ],
            gap=1,
        )
        preset_gif_progress_block = None

    mo.output.replace(
        _build_preset_sections(
            _build_preset_gif_accordion(
                progress_block=preset_gif_progress_block,
                result_block=preset_gif_result_block,
            )
        )
    )
    return preset_open_gif_folder_button, preset_gif_saved_path


@app.cell
def _(mo, open_path_in_file_browser, preset_gif_saved_path, preset_open_gif_folder_button):
    if preset_open_gif_folder_button is not None:
        mo.stop(not preset_open_gif_folder_button.value)
        if preset_gif_saved_path is not None:
            open_path_in_file_browser(preset_gif_saved_path)
    return


if __name__ == "__main__":
    app.run()
