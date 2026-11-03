"""# 🖥️ CLI Commands

CLI command definitions for dataset generation workflows.
"""

# ruff: noqa: FBT002

from collections import defaultdict
from typing import Annotated

import typer
from loguru import logger
from rich.console import Console

from endlesswm.cli.generation import run_generate_request
from endlesswm.pipeline.config.loader import (
    list_all_drift_presets,
    list_available_components,
    load_component_config,
    load_drift_preset,
)
from endlesswm.pipeline.config.sweeps import parse_string_sweep
from endlesswm.pipeline.drift_capabilities import get_drift_capability_report

app = typer.Typer(help="♾️🌊🗺️ Endless Wafer Maps")
console = Console()


def _run_request_or_exit(**request_kwargs: object) -> None:
    """🧯 Run generation request and map known failures to CLI exit codes."""
    try:
        run_generate_request(**request_kwargs)
    except FileNotFoundError as exc:
        logger.error("{}", exc)
        logger.info("Available drift presets: {}", ", ".join(list_all_drift_presets()))
        raise typer.Exit(1) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("{}", exc)
        raise typer.Exit(1) from exc


def _summarize_preset_capabilities(preset_name: str) -> tuple[str, str]:
    """🧪 Summarize drift support status for one preset."""
    preset = load_drift_preset(preset_name, validate_runtime_capabilities=False)
    report = get_drift_capability_report(preset.drifts)
    experimental = [axis for axis, meta in report.items() if not meta["supported_by_runtime"]]
    return ("experimental", ", ".join(experimental)) if experimental else ("ready", "-")


def _split_preset_group(preset_name: str) -> tuple[str, str]:
    """Split a preset path into group name and local preset name."""
    return tuple(preset_name.split("/", 1)) if "/" in preset_name else ("ungrouped", preset_name)  # type: ignore[return-value]


def _format_preset_size_summary(
    *,
    num_tasks: int,
    num_samples_per_task: int,
    total_samples: int,
    num_samples_first_task: int | None,
) -> str:
    """Build a readable one-line size summary for one preset."""
    first_task = num_samples_first_task or num_samples_per_task
    if first_task == num_samples_per_task:
        return f"{num_tasks:,} tasks, {num_samples_per_task:,} per task, {total_samples:,} total"

    return f"{num_tasks:,} tasks, {first_task:,} in task 0, {num_samples_per_task:,} in later tasks, {total_samples:,} total"


def _run_bulk_generate_requests(
    preset_names: list[str],
    *,
    dry_run: bool,
    validate_only: bool,
    show_resolved_config: bool,
    show_capabilities: bool,
    allow_unsupported_drifts: bool,
    skip_existing: bool = False,
) -> None:
    """Run generation sequentially for multiple presets and summarize failures."""
    failures: list[tuple[str, str]] = []

    for preset_name in preset_names:
        try:
            run_generate_request(
                preset=preset_name,
                dry_run=dry_run,
                validate_only=validate_only,
                show_resolved_config=show_resolved_config,
                show_capabilities=show_capabilities,
                allow_unsupported_drifts=allow_unsupported_drifts,
                skip_existing=skip_existing,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append((preset_name, str(exc)))
            logger.error("Preset '{}' failed: {}", preset_name, exc)

    success_count = len(preset_names) - len(failures)
    logger.info(
        "Bulk generation finished: {} succeeded, {} failed (total={}).",
        success_count,
        len(failures),
        len(preset_names),
    )

    if failures:
        raise typer.Exit(1)


@app.command()
def generate(
    preset: str | None = typer.Argument(
        None,
        help="Preset identifier: full relative path (e.g. 'evaluation/e10k') or unique basename (e.g. 'e10k').",
    ),
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show resolved runs without generating")] = False,
    validate_only: Annotated[
        bool,
        typer.Option("--validate-only", help="Validate full runtime configuration without exporting"),
    ] = False,
    show_resolved_config: Annotated[
        bool,
        typer.Option("--show-resolved-config", help="Print resolved runtime config for each run"),
    ] = False,
    show_capabilities: Annotated[
        bool,
        typer.Option(
            "--show-capabilities",
            help="Print runtime support status for the selected drift strategies in dry-run and show-resolved-config modes.",
        ),
    ] = False,
    allow_unsupported_drifts: Annotated[
        bool,
        typer.Option(
            "--allow-unsupported-drifts",
            help="Skip wrapper capability checks to prototype new drift strategies.",
        ),
    ] = False,
    all_presets: Annotated[
        bool,
        typer.Option("--all-presets", help="Generate all available presets."),
    ] = False,
    skip_existing: Annotated[
        bool,
        typer.Option(
            "--skip-existing",
            help="Skip cells whose resolved output file already exists. Useful for resuming or syncing across machines.",
        ),
    ] = False,
    preset_group: str | None = typer.Option(
        None,
        "--preset-group",
        help="Generate all presets in one group (folder under drift_presets, e.g. 'coverage').",
    ),
) -> None:
    """🎲 Generate datasets from a drift preset (or a group / all presets).

    All run customization (sweeps, drift parameters, stream size, components)
    comes from the preset YAML; this command only selects which preset(s) to run.
    """
    if all_presets and preset_group is not None:
        logger.error("--all-presets and --preset-group are mutually exclusive.")
        raise typer.Exit(1)
    if all_presets and preset is not None:
        logger.error("Pass either a preset positional argument or --all-presets, not both.")
        raise typer.Exit(1)
    if preset_group is not None and preset is not None:
        logger.error("Pass either a preset positional argument or --preset-group, not both.")
        raise typer.Exit(1)

    all_preset_names = list_all_drift_presets()

    if all_presets:
        if not all_preset_names:
            logger.error("No presets found.")
            raise typer.Exit(1)
        _run_bulk_generate_requests(
            preset_names=all_preset_names,
            dry_run=dry_run,
            validate_only=validate_only,
            show_resolved_config=show_resolved_config,
            show_capabilities=show_capabilities,
            allow_unsupported_drifts=allow_unsupported_drifts,
            skip_existing=skip_existing,
        )
        return

    if preset_group is not None:
        group_prefix = f"{preset_group}/"
        grouped_presets = [name for name in all_preset_names if name.startswith(group_prefix)]
        if not grouped_presets:
            logger.error("No presets found for group '{}'.", preset_group)
            logger.info(
                "Available groups: {}",
                ", ".join(sorted({name.split("/", 1)[0] for name in all_preset_names if "/" in name})),
            )
            raise typer.Exit(1)
        _run_bulk_generate_requests(
            preset_names=grouped_presets,
            dry_run=dry_run,
            validate_only=validate_only,
            show_resolved_config=show_resolved_config,
            show_capabilities=show_capabilities,
            allow_unsupported_drifts=allow_unsupported_drifts,
            skip_existing=skip_existing,
        )
        return

    if preset is None:
        logger.error("Provide a preset, or pass --all-presets / --preset-group.")
        raise typer.Exit(1)

    _run_request_or_exit(
        preset=preset,
        dry_run=dry_run,
        validate_only=validate_only,
        show_resolved_config=show_resolved_config,
        show_capabilities=show_capabilities,
        allow_unsupported_drifts=allow_unsupported_drifts,
        skip_existing=skip_existing,
    )


@app.command("list-presets")
def list_presets() -> None:
    """📋 List all available dataset presets."""
    presets = list_all_drift_presets()

    if not presets:
        logger.info("No presets found.")
        return

    sorted_presets = sorted(presets)
    grouped_presets: dict[str, list[str]] = defaultdict(list)
    for preset_name in sorted_presets:
        group_name, _ = _split_preset_group(preset_name)
        grouped_presets[group_name].append(preset_name)

    console.print(
        f"Available drift presets: [bold]{len(sorted_presets)}[/bold] presets in [bold]{len(grouped_presets)}[/bold] groups",
    )

    console.print()
    for group_name in sorted(grouped_presets):
        console.print(f"[bold cyan]{group_name}[/bold cyan] ([dim]{len(grouped_presets[group_name])} presets[/dim])")
        console.print()

        for preset_name in grouped_presets[group_name]:
            _, local_name = _split_preset_group(preset_name)
            try:
                preset = load_drift_preset(preset_name, validate_runtime_capabilities=False)
                display_name = local_name
                _drift_status, experimental_axes = _summarize_preset_capabilities(preset_name)
                headline = f"• [bold]{display_name}[/bold]"

                console.print(headline)
                console.print(f"  key: {local_name}")

                sweeps = preset.sweeps
                stream_size_sweep = sweeps.stream_size if sweeps is not None else None
                if stream_size_sweep is not None:
                    for stream_size_value in parse_string_sweep(stream_size_sweep, "stream_size"):
                        size_config = load_component_config("stream_size", stream_size_value)
                        num_tasks = size_config.num_tasks
                        num_samples_per_task = size_config.num_samples_per_task
                        num_samples_first_task = size_config.num_samples_first_task
                        total = num_samples_per_task * num_tasks + ((num_samples_first_task or num_samples_per_task) - num_samples_per_task)
                        console.print(
                            f"  size ({stream_size_value}): "
                            + _format_preset_size_summary(
                                num_tasks=num_tasks,
                                num_samples_per_task=num_samples_per_task,
                                total_samples=total,
                                num_samples_first_task=num_samples_first_task,
                            ),
                        )
                else:
                    console.print(
                        "  size: "
                        + _format_preset_size_summary(
                            num_tasks=preset.num_tasks,
                            num_samples_per_task=preset.num_samples_per_task,
                            total_samples=preset.total_samples,
                            num_samples_first_task=preset.num_samples_first_task,
                        ),
                    )
                if experimental_axes != "-":
                    console.print(f"  needs work on: {experimental_axes}")
                console.print(f"  summary: {preset.description or '-'}")
            except (FileNotFoundError, KeyError, TypeError, ValueError):
                console.print(f"• [bold]{local_name}[/bold]  [red]error[/red]")
                console.print(f"  key: {local_name}")

        console.print()


@app.command("list-components")
def list_components() -> None:
    """🧩 List valid override values in copy-paste friendly CLI format."""
    stems = "|".join(list_available_components("stem"))
    branches = "|".join(list_available_components("branches"))
    cardinality_priors = "|".join(list_available_components("cardinality_priors"))
    class_priors = "|".join(list_available_components("class_priors"))
    postprocessing = "|".join(list_available_components("postprocessing"))
    shapes = "|".join(list_available_components("shapes"))

    console.print("Available component options:")
    console.print(f"  --stem {stems}")
    console.print(f"  --branches {branches}")
    console.print(f"  --cardinality-prior {cardinality_priors}")
    console.print(f"  --class-prior {class_priors}")
    console.print(f"  --postprocessing {postprocessing}")
    console.print(f"  --shapes {shapes}")


if __name__ == "__main__":
    app()
