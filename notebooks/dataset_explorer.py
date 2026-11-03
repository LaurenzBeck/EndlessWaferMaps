# ruff: noqa: ANN, B018, E402, INP001, I001

"""💾 Marimo dataset explorer."""

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full", app_title="🌌 Dataset Explorer")


@app.cell
def _():
    import h5py
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from pathlib import Path

    from endlesswm.explorer_support import (
        build_default_gif_path_from_dataset_path,
        build_explorer_gif_path,
        open_path_in_file_browser,
        render_sample_grid_frame,
        save_sample_grid_gif,
    )
    from endlesswm.io.datasets import H5WM
    from endlesswm.patterns import classes as DEFAULT_CLASS_NAMES
    from endlesswm.pipeline.config.loader import load_drift_preset
    from endlesswm.cli.generation import build_tree_from_preset
    from endlesswm.visualizations import (
        build_parameter_evolution_heatmap,
        build_parameter_evolution_matrix,
        build_single_parameter_schedule_figure,
        compact_heatmap,
        plot_cardinality_distribution_over_tasks,
        plot_class_distribution_over_tasks,
        plot_mask,
        plot_snr_over_stream,
        resolve_scheduled_parameter_name,
        resolve_numeric_parameter_names,
    )

    return (
        DEFAULT_CLASS_NAMES,
        H5WM,
        Path,
        build_default_gif_path_from_dataset_path,
        build_explorer_gif_path,
        build_parameter_evolution_heatmap,
        build_parameter_evolution_matrix,
        build_single_parameter_schedule_figure,
        build_tree_from_preset,
        compact_heatmap,
        h5py,
        load_drift_preset,
        mo,
        np,
        open_path_in_file_browser,
        pd,
        plot_cardinality_distribution_over_tasks,
        plot_class_distribution_over_tasks,
        plot_mask,
        plot_snr_over_stream,
        plt,
        render_sample_grid_frame,
        resolve_numeric_parameter_names,
        resolve_scheduled_parameter_name,
        save_sample_grid_gif,
    )


@app.cell
def _(
    DEFAULT_CLASS_NAMES,
    H5WM,
    build_parameter_evolution_heatmap,
    build_parameter_evolution_matrix,
    build_single_parameter_schedule_figure,
    build_tree_from_preset,
    compact_heatmap,
    h5py,
    load_drift_preset,
    np,
    pd,
    plot_cardinality_distribution_over_tasks,
    plot_class_distribution_over_tasks,
    plot_mask,
    plot_snr_over_stream,
    plt,
    resolve_scheduled_parameter_name,
):
    def read_dataset_metadata(file_path: str) -> dict[str, object]:
        with h5py.File(file_path, "r") as handle:
            task_keys = sorted(handle["tasks"].keys(), key=int)
            meta = handle.get("meta")
            class_names = list(DEFAULT_CLASS_NAMES)
            preset_identifier = None
            if meta is not None:
                if "classes" in meta:
                    class_names = [str(value.decode("utf-8") if isinstance(value, bytes) else value) for value in meta["classes"][...]]
                preset_identifier = meta.attrs.get("preset_identifier")
            return {
                "num_tasks": len(task_keys),
                "class_names": class_names,
                "preset_identifier": str(preset_identifier) if preset_identifier else None,
            }

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

    def build_parameter_heatmap(parameter_df: pd.DataFrame, group_parameter_names: list[str]):
        matrix = build_parameter_evolution_matrix(
            parameter_df,
            parameter_names=group_parameter_names,
            statistic="median",
            include_constant_parameters=False,
            excluded_name_fragments=("polarity",),
        )
        return build_parameter_evolution_heatmap(
            matrix,
            title="",
            normalize_rows=True,
            normalize_with_full_allowed_range=True,
            show_parameter_labels=False,
        )

    def build_dataset_bounds(parameter_df: pd.DataFrame, preset_identifier: str, parameter_name: str, num_tasks: int) -> pd.DataFrame:
        preset = load_drift_preset(preset_identifier, validate_runtime_capabilities=False)
        preset.num_tasks = num_tasks
        tree = build_tree_from_preset(preset)
        schedule_df = tree.params.to_dataframe(number_of_update_steps=max(num_tasks - 1, 0))
        observed_df = (
            parameter_df.groupby("task", sort=True)[parameter_name]
            .agg(observed_min="min", observed_median="median", observed_max="max")
            .reset_index()
        )

        schedule_parameter_name = resolve_scheduled_parameter_name(schedule_df, parameter_name)
        if schedule_parameter_name is None:
            return observed_df

        scheduled_rows = []
        for task_index, bounds in enumerate(schedule_df[f"{schedule_parameter_name}_r"]):
            scheduled_rows.append(
                {
                    "task": task_index,
                    "scheduled_min": float(bounds[0]),
                    "scheduled_max": float(bounds[1]),
                },
            )

        return pd.DataFrame(scheduled_rows).merge(observed_df, on="task", how="left")

    def build_single_parameter_schedule_section(
        parameter_df: pd.DataFrame,
        parameter_name: str,
        metadata: dict[str, object] | None,
    ):
        if metadata is None or not metadata.get("preset_identifier"):
            bounds_df = (
                parameter_df.groupby("task", sort=True)[parameter_name]
                .agg(observed_min="min", observed_median="median", observed_max="max")
                .reset_index()
            )
        else:
            bounds_df = build_dataset_bounds(
                parameter_df,
                str(metadata["preset_identifier"]),
                parameter_name,
                int(metadata["num_tasks"]),
            )
        return build_single_parameter_schedule_figure(bounds_df, parameter_name)

    def build_distribution_section(mo, file_path: str):
        dataset = H5WM(file_path)
        return mo.vstack([plot_class_distribution_over_tasks(dataset), plot_cardinality_distribution_over_tasks(dataset)])

    def build_snr_section(file_path: str, snr_kind: str, db: bool, max_samples_per_task: int):
        fig = plot_snr_over_stream(
            file_path,
            snr_kind=snr_kind,
            db=db,
            max_samples_per_task=max_samples_per_task,
        )
        plt.close(fig)
        return fig

    def sample_task_arrays(file_path: str, task_index: int, sample_view: str, num_samples: int = 36) -> list[np.ndarray]:
        with h5py.File(file_path, "r") as handle:
            task_group = handle["tasks"][str(task_index)]
            field_name = "masks" if sample_view == "mask annotations" else sample_view
            sample_ds = task_group[field_name]
            indices = np.linspace(0, sample_ds.shape[0] - 1, min(num_samples, sample_ds.shape[0]), dtype=int)
            return [sample_ds[int(index)] for index in indices]

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
        build_distribution_section,
        build_parameter_heatmap,
        build_sample_gallery,
        build_single_parameter_schedule_section,
        build_snr_section,
        make_parameter_groups,
        read_dataset_metadata,
        sample_task_arrays,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 🌌 EndlessWM - Dataset Explorer
    """)
    return


@app.cell
def _(Path, mo):
    initial_path = Path("experiments/data/generated")
    dataset_widget = mo.ui.file_browser(
        initial_path=initial_path if initial_path.exists() else Path.cwd(),
        filetypes=[".hdf5"],
        selection_mode="file",
        multiple=False,
        label="💾 select a dataset:",
    )
    return (dataset_widget,)


@app.cell
def _(first_group, grouped_parameter_names, mo):
    heatmap_group_widget = None
    if grouped_parameter_names and first_group is not None:
        heatmap_group_widget = mo.ui.dropdown({name: name for name in grouped_parameter_names}, value=first_group, label="🧭 group")
    return (heatmap_group_widget,)


@app.cell
def _(first_group, grouped_parameter_names, mo):
    schedule_group_widget = None
    if grouped_parameter_names and first_group is not None:
        schedule_group_widget = mo.ui.dropdown({name: name for name in grouped_parameter_names}, value=first_group, label="🧭 group")
    return (schedule_group_widget,)


@app.cell
def _(grouped_parameter_names, mo, schedule_group_widget):
    schedule_parameter_widget = None
    if schedule_group_widget is not None:
        selected_group_parameter_names = grouped_parameter_names[schedule_group_widget.value]
        schedule_parameter_widget = mo.ui.dropdown(
            {name: name for name in selected_group_parameter_names},
            value=selected_group_parameter_names[0],
            label="🎚️ param",
        )
    return (schedule_parameter_widget,)


@app.cell
def _(metadata, mo):
    sample_view_widget = None
    task_slider_widget = None
    if metadata is not None:
        _max_task_index = max(0, metadata["num_tasks"] - 1)
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
            label="🌐 view",
        )
    return sample_view_widget, task_slider_widget


@app.cell
def _(
    build_default_gif_path_from_dataset_path,
    metadata,
    mo,
    selected_dataset_path,
):
    dataset_gif_form = None
    if metadata is not None and selected_dataset_path:
        _max_task_index = max(0, int(metadata["num_tasks"]) - 1)
        _default_output_path = build_default_gif_path_from_dataset_path(selected_dataset_path).as_posix()

        def _validate_dataset_gif_form(values):
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

        dataset_gif_form = (
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
                validate=_validate_dataset_gif_form,
            )
        )
    return (dataset_gif_form,)


@app.cell
def _(mo):
    get_dataset_gif_state, set_dataset_gif_state = mo.state({"request": None, "saved_path": None})
    return get_dataset_gif_state, set_dataset_gif_state


@app.cell
def _(dataset_widget):
    selected_path = dataset_widget.path()
    selected_dataset_path = str(selected_path) if selected_path is not None else None
    return (selected_dataset_path,)


@app.cell
def _(
    build_parameter_heatmap,
    grouped_parameter_names,
    heatmap_group_widget,
    metadata,
    mo,
    parameter_df,
):
    heatmap_accordion = None
    if metadata is not None and parameter_df is not None and grouped_parameter_names and heatmap_group_widget is not None:
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
    metadata,
    mo,
    parameter_df,
    schedule_group_widget,
    schedule_parameter_widget,
):
    single_parameter_accordion = None
    if metadata is not None and parameter_df is not None and schedule_group_widget is not None and schedule_parameter_widget is not None:
        schedule_controls = mo.hstack([schedule_group_widget, schedule_parameter_widget], justify="start", wrap=True)
        schedule_body = mo.vstack(
            [
                schedule_controls,
                mo.lazy(
                    lambda: build_single_parameter_schedule_section(
                        parameter_df,
                        schedule_parameter_widget.value,
                        metadata,
                    ),
                    show_loading_indicator=True,
                ),
            ],
            gap=1,
        )
        single_parameter_accordion = mo.accordion({"🎚️ single parameter schedules": schedule_body})
    return (single_parameter_accordion,)


@app.cell
def _(build_distribution_section, metadata, mo, selected_dataset_path):
    distribution_accordion = None
    if metadata is not None and selected_dataset_path:
        distribution_body = mo.lazy(
            lambda: build_distribution_section(mo, selected_dataset_path),
            show_loading_indicator=True,
        )
        distribution_accordion = mo.accordion({"🏷️ label distributions": distribution_body})
    return (distribution_accordion,)


@app.cell
def _(metadata, mo):
    snr_kind_widget = None
    snr_db_widget = None
    snr_max_samples_widget = None
    if metadata is not None:
        snr_kind_widget = mo.ui.dropdown(
            {"energy": "energy", "variance": "variance"},
            value="energy",
            label="📡 SNR kind",
        )
        snr_db_widget = mo.ui.switch(value=True, label="dB")
        snr_max_samples_widget = mo.ui.slider(
            start=20,
            stop=1000,
            step=20,
            value=200,
            label="🔢 max samples / task",
            include_input=True,
            show_value=True,
        )
    return snr_db_widget, snr_kind_widget, snr_max_samples_widget


@app.cell
def _(
    build_snr_section,
    metadata,
    mo,
    selected_dataset_path,
    snr_db_widget,
    snr_kind_widget,
    snr_max_samples_widget,
):
    snr_accordion = None
    if (
        metadata is not None
        and selected_dataset_path
        and snr_kind_widget is not None
        and snr_db_widget is not None
        and snr_max_samples_widget is not None
    ):
        snr_controls = mo.hstack(
            [snr_kind_widget, snr_db_widget, snr_max_samples_widget],
            justify="start",
            wrap=True,
        )
        snr_body = mo.vstack(
            [
                snr_controls,
                mo.lazy(
                    lambda: build_snr_section(
                        selected_dataset_path,
                        snr_kind_widget.value,
                        bool(snr_db_widget.value),
                        int(snr_max_samples_widget.value),
                    ),
                    show_loading_indicator=True,
                ),
            ],
            gap=1,
        )
        snr_accordion = mo.accordion({"📡 SNR over stream": snr_body})
    return (snr_accordion,)


@app.cell
def _(
    build_sample_gallery,
    mo,
    sample_task_arrays,
    sample_view_widget,
    selected_dataset_path,
    task_slider_widget,
):
    samples_accordion = None
    if sample_view_widget is not None and task_slider_widget is not None and selected_dataset_path:
        sample_controls = mo.hstack([task_slider_widget, sample_view_widget], justify="start", wrap=True)
        sample_body = mo.vstack(
            [
                sample_controls,
                mo.lazy(
                    lambda: build_sample_gallery(
                        mo,
                        sample_task_arrays(selected_dataset_path, int(task_slider_widget.value), sample_view_widget.value, 36),
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
    dataset_gif_form,
    dataset_widget,
    distribution_accordion,
    get_dataset_gif_state,
    h5py,
    heatmap_accordion,
    metadata,
    mo,
    np,
    render_sample_grid_frame,
    samples_accordion,
    save_sample_grid_gif,
    selected_dataset_path,
    set_dataset_gif_state,
    single_parameter_accordion,
    snr_accordion,
):
    def _build_dataset_sections(dataset_gif_accordion):
        sections = [dataset_widget]
        for section in [
            heatmap_accordion,
            single_parameter_accordion,
            distribution_accordion,
            snr_accordion,
            samples_accordion,
            dataset_gif_accordion,
        ]:
            if section is not None:
                sections.append(section)
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

    def _build_dataset_gif_accordion(progress_block=None, result_block=None):
        if metadata is None or dataset_gif_form is None:
            return None

        gif_sections = [dataset_gif_form]
        if progress_block is not None:
            gif_sections.append(progress_block)
        if result_block is not None:
            gif_sections.append(result_block)
        return mo.accordion({"🎞️ animated sample GIF": mo.vstack(gif_sections, gap=1)})

    dataset_gif_result_block = None
    dataset_gif_progress_block = None
    dataset_gif_state = get_dataset_gif_state()
    dataset_gif_request = None
    dataset_gif_saved_path = None
    dataset_open_gif_folder_button = None

    if metadata is not None and dataset_gif_form is not None and dataset_gif_form.value is not None and selected_dataset_path:
        _gif_settings = dataset_gif_form.value
        _start_task = int(_gif_settings["start_task"])
        _end_task = int(_gif_settings["end_task"])
        _task_stride = int(_gif_settings["task_stride"])
        _rows = int(_gif_settings["rows"])
        _columns = int(_gif_settings["columns"])
        _sample_view = str(_gif_settings["sample_view"])
        _frame_duration_ms = int(_gif_settings["frame_duration_ms"])
        _samples_per_frame = _rows * _columns
        _task_indices = list(range(_start_task, _end_task + 1, _task_stride))
        _output_path = build_explorer_gif_path(
            str(_gif_settings["output_path"]),
            default_output_path=build_default_gif_path_from_dataset_path(selected_dataset_path),
        )
        dataset_gif_request = {
            "dataset_path": selected_dataset_path,
            "sample_view": _sample_view,
            "start_task": _start_task,
            "end_task": _end_task,
            "task_stride": _task_stride,
            "rows": _rows,
            "columns": _columns,
            "frame_duration_ms": _frame_duration_ms,
            "output_path": _output_path.as_posix(),
        }

        if dataset_gif_state.get("request") == dataset_gif_request and dataset_gif_state.get("saved_path"):
            dataset_gif_saved_path = str(dataset_gif_state["saved_path"])
        else:
            dataset_gif_progress_block = _build_progress_block(
                "Creating GIF",
                "Preparing frame rendering",
                0,
                len(_task_indices),
            )
            mo.output.replace(_build_dataset_sections(_build_dataset_gif_accordion(progress_block=dataset_gif_progress_block)))

            _frames = []
            with h5py.File(selected_dataset_path, "r") as handle:
                _field_name = "masks" if _sample_view == "mask annotations" else _sample_view
                for frame_index, task_index in enumerate(_task_indices, start=1):
                    dataset_gif_progress_block = _build_progress_block(
                        "Rendering frames",
                        f"Rendering frame {frame_index} of {len(_task_indices)}",
                        frame_index - 1,
                        len(_task_indices),
                    )
                    mo.output.replace(_build_dataset_sections(_build_dataset_gif_accordion(progress_block=dataset_gif_progress_block)))
                    _sample_ds = handle["tasks"][str(task_index)][_field_name]
                    _sample_indices = np.linspace(
                        0,
                        _sample_ds.shape[0] - 1,
                        min(_samples_per_frame, _sample_ds.shape[0]),
                        dtype=int,
                    )
                    _samples = [_sample_ds[int(sample_index)] for sample_index in _sample_indices]
                    _frames.append(
                        render_sample_grid_frame(
                            _samples,
                            _sample_view,
                            _rows,
                            _columns,
                        )
                    )

            dataset_gif_progress_block = _build_progress_block(
                "Saving GIF",
                f"Writing {_output_path.name}",
                len(_task_indices),
                len(_task_indices),
            )
            mo.output.replace(_build_dataset_sections(_build_dataset_gif_accordion(progress_block=dataset_gif_progress_block)))
            dataset_gif_saved_path = save_sample_grid_gif(_frames, _output_path, _frame_duration_ms).as_posix()
            set_dataset_gif_state({"request": dataset_gif_request, "saved_path": dataset_gif_saved_path})

        dataset_open_gif_folder_button = mo.ui.run_button(label="Open folder")

        dataset_gif_result_block = mo.vstack(
            [
                mo.md(f"Saved GIF to `{dataset_gif_saved_path}`"),
                dataset_open_gif_folder_button,
                mo.md(
                    f"Created {len(_task_indices)} frame(s) with a {_rows}x{_columns} grid and {_samples_per_frame} sample(s) per frame."
                ),
                mo.image(src=dataset_gif_saved_path),
            ],
            gap=1,
        )
        dataset_gif_progress_block = None

    mo.output.replace(
        _build_dataset_sections(
            _build_dataset_gif_accordion(
                progress_block=dataset_gif_progress_block,
                result_block=dataset_gif_result_block,
            )
        )
    )
    return dataset_gif_saved_path, dataset_open_gif_folder_button


@app.cell
def _(
    dataset_gif_saved_path,
    dataset_open_gif_folder_button,
    mo,
    open_path_in_file_browser,
):
    if dataset_open_gif_folder_button is not None:
        mo.stop(not dataset_open_gif_folder_button.value)
        if dataset_gif_saved_path is not None:
            open_path_in_file_browser(dataset_gif_saved_path)
    return


@app.cell
def _(H5WM, read_dataset_metadata, selected_dataset_path):
    metadata = None
    parameter_df = None
    if selected_dataset_path:
        metadata = read_dataset_metadata(selected_dataset_path)
        parameter_df = H5WM(selected_dataset_path).parameters
    return metadata, parameter_df


@app.cell
def _(
    DEFAULT_CLASS_NAMES,
    make_parameter_groups,
    metadata,
    parameter_df,
    resolve_numeric_parameter_names,
):
    first_group = None
    grouped_parameter_names = {}
    if metadata is not None and parameter_df is not None:
        parameter_names, _ = resolve_numeric_parameter_names(
            parameter_df,
            include_constant_parameters=True,
            excluded_name_fragments=("polarity",),
        )
        grouped_parameter_names = make_parameter_groups(parameter_names, metadata.get("class_names", DEFAULT_CLASS_NAMES))
        first_group = next(iter(grouped_parameter_names))
    return first_group, grouped_parameter_names


if __name__ == "__main__":
    app.run()
