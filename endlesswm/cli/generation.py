"""# ⚙️ CLI Generation Runtime

Resolve a preset (with inline sweeps) into a list of concrete runs and
either dry-run, validate, or execute them.

Selection happens entirely via preset YAML. The CLI does not accept ad-hoc
component overrides; every dataset is described by a preset file under
``endlesswm/configs/drift_presets/<group>/<alias>.yaml``.
"""

# ruff: noqa: ANN401, EM101, EM102, TRY003

import json
import shutil
from itertools import product
from pathlib import Path
from typing import Any

from loguru import logger
from streamgen.samplers.tree import SamplingTree
from tqdm.auto import tqdm

from endlesswm.io.export import generate_and_export_dataset
from endlesswm.pipeline.config.loader import (
    DRIFT_PRESETS_DIR,
    load_component_config,
    load_drift_preset,
    resolve_drift_preset_name,
)
from endlesswm.pipeline.config.naming import (
    preset_alias,
    resolve_output_path,
)
from endlesswm.pipeline.config.schema import DatasetPreset
from endlesswm.pipeline.config.sweeps import parse_float_sweep, parse_int_sweep, parse_string_sweep
from endlesswm.pipeline.drift_capabilities import get_drift_capability_report, validate_drift_runtime_capabilities
from endlesswm.pipeline.tree_factory import TreeFactory


def _apply_stream_size_value(run_preset: DatasetPreset, stream_size_name: str) -> None:
    size_cfg = load_component_config("stream_size", stream_size_name)
    run_preset.stream_size = stream_size_name
    run_preset.num_tasks = size_cfg.num_tasks
    run_preset.num_samples_per_task = size_cfg.num_samples_per_task
    run_preset.num_samples_first_task = size_cfg.num_samples_first_task
    run_preset.num_validation_samples = size_cfg.num_validation_samples


def _paired_drift_sweep_rows(base_preset: DatasetPreset) -> list[dict[str, float]]:
    """Build paired fraction/sigma rows without Cartesian products."""
    sweeps = base_preset.sweeps
    if sweeps is None:
        return [{}]

    paired: dict[str, list[float]] = {}
    if sweeps.fraction is not None:
        paired["fraction"] = parse_float_sweep(sweeps.fraction, "fraction")
    if sweeps.sigma is not None:
        paired["sigma"] = parse_float_sweep(sweeps.sigma, "sigma")

    if not paired:
        return [{}]

    lengths = {len(values) for values in paired.values()}
    max_length = max(lengths)
    if any(length not in {1, max_length} for length in lengths):
        raise ValueError("Preset paired sweeps must have equal length or length 1 for broadcasting.")

    return [{key: values[idx if len(values) > 1 else 0] for key, values in paired.items()} for idx in range(max_length)]


def _expand_preset_sweeps(base_preset: DatasetPreset) -> list[DatasetPreset]:
    """Expand inline sweeps in the preset into concrete run presets."""
    sweeps = base_preset.sweeps

    seed_values = parse_int_sweep(sweeps.seed, "seed") if sweeps and sweeps.seed else [base_preset.seed]
    task_values = parse_int_sweep(sweeps.tasks, "tasks") if sweeps and sweeps.tasks else [base_preset.num_tasks]
    sample_values = parse_int_sweep(sweeps.samples, "samples") if sweeps and sweeps.samples else [base_preset.num_samples_per_task]
    stream_size_values = (
        parse_string_sweep(sweeps.stream_size, "stream_size") if sweeps and sweeps.stream_size else [base_preset.stream_size]
    )
    paired_rows = _paired_drift_sweep_rows(base_preset)

    override_tasks = bool(sweeps and sweeps.tasks)
    override_samples = bool(sweeps and sweeps.samples)

    run_presets: list[DatasetPreset] = []
    for task_val, sample_val, seed_val, stream_size_name, paired_row in product(
        task_values,
        sample_values,
        seed_values,
        stream_size_values,
        paired_rows,
    ):
        run_preset = base_preset.model_copy(deep=True)
        if stream_size_name and stream_size_name != run_preset.stream_size:
            _apply_stream_size_value(run_preset, stream_size_name)
        if override_tasks:
            run_preset.num_tasks = task_val
        if override_samples:
            run_preset.num_samples_per_task = sample_val
        run_preset.seed = seed_val
        for key in ("fraction", "sigma"):
            if key in paired_row:
                run_preset.drifts.covariate.params[key] = paired_row[key]
                run_preset.drifts.concept.params[key] = paired_row[key]
        run_presets.append(run_preset)

    return run_presets


def _make_validation_preset(run_preset: DatasetPreset, num_samples: int) -> DatasetPreset:
    """🧪 Build paired validation preset with uniform priors and fixed seed."""
    val = run_preset.model_copy(deep=True)
    val.seed = 103
    val.num_samples_per_task = num_samples
    val.num_samples_first_task = num_samples
    val.num_validation_samples = None
    val.cardinality_prior = "uniform"
    val.class_prior = "uniform"
    val.drifts.prior.cardinality_schedule = "fixed"
    val.drifts.prior.class_schedule = "fixed"
    val.drifts.prior.params = {}
    val.sweeps = None
    return val


def _plan_runs(base_preset: DatasetPreset) -> list[tuple[DatasetPreset, Path]]:
    """Return ordered ``(preset, output_path)`` pairs, appending validation sets where configured."""
    plan = [(main, resolve_output_path(main)) for main in _expand_preset_sweeps(base_preset)]
    for main, main_path in list(plan):
        if main.num_validation_samples:
            val_path = main_path.with_name(f"{main_path.stem}-t{main_path.suffix}")
            plan.append((_make_validation_preset(main, main.num_validation_samples), val_path))
    return plan


def _validate_unique_output_paths(plan: list[tuple[DatasetPreset, Path]]) -> None:
    seen: dict[Path, list[int]] = {}
    for run_preset, output_path in plan:
        seen.setdefault(output_path, []).append(run_preset.seed)
    conflicts = {path: seeds for path, seeds in seen.items() if len(seeds) > 1}
    if conflicts:
        examples = "; ".join(f"{path} (seeds={seeds})" for path, seeds in conflicts.items())
        raise ValueError(f"Multiple runs resolve to the same output path. Conflicts: {examples}")


def build_tree_from_preset(preset_config: DatasetPreset) -> SamplingTree:
    """🌳 Create a fresh sampling tree for a preset."""
    return TreeFactory.from_configs(
        stem_config=load_component_config("stem", preset_config.stem),
        branches_config=load_component_config("branches", preset_config.branches),
        cardinality_prior_config=load_component_config("cardinality_priors", preset_config.cardinality_prior),
        class_prior_config=load_component_config("class_priors", preset_config.class_prior),
        postprocessing_config=load_component_config("postprocessing", preset_config.postprocessing),
        shape_config=load_component_config("shapes", preset_config.shapes),
        drift_config=preset_config.drifts,
        num_tasks=preset_config.num_tasks,
        seed=preset_config.seed,
    )


def _log_resolved_config(preset_config: DatasetPreset, output_path: Path) -> None:
    payload = preset_config.model_dump(mode="json") | {"resolved_output_path": str(output_path)}
    logger.info("Resolved config for seed {}:\n{}", preset_config.seed, json.dumps(payload, indent=2, sort_keys=True))


def _copy_preset_alongside_outputs(base_preset: DatasetPreset, plan: list[tuple[DatasetPreset, Path]]) -> None:
    """Copy the source preset YAML once per output directory."""
    if not base_preset.preset_id:
        return
    source_path = DRIFT_PRESETS_DIR / f"{base_preset.preset_id}.yaml"
    if not source_path.exists():
        logger.warning("Could not find source preset YAML to copy: {}", source_path)
        return

    target_filename = f"{preset_alias(base_preset)}.yaml"
    for target_dir in {output_path.parent for _, output_path in plan}:
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_dir / target_filename)


def _run_generation_flow(
    plan: list[tuple[DatasetPreset, Path]],
    *,
    command_invocation: str,
    dry_run: bool,
    validate_only: bool,
    show_resolved_config: bool,
    show_capabilities: bool,
    skip_existing: bool,
) -> None:
    skipped = 0
    for run_preset, output_path in tqdm(plan, desc="Runs", disable=len(plan) <= 1):
        if skip_existing and output_path.exists():
            logger.info("Skipping existing dataset: {}", output_path)
            skipped += 1
            continue

        if show_capabilities and (dry_run or show_resolved_config):
            report = get_drift_capability_report(run_preset.drifts)
            logger.info("Drift capability report for seed {}:\n{}", run_preset.seed, json.dumps(report, indent=2, sort_keys=True))

        if dry_run:
            logger.info(
                "Dry-run seed={} tasks={} samples={} output={}",
                run_preset.seed,
                run_preset.num_tasks,
                run_preset.num_samples_per_task,
                output_path,
            )
            if show_resolved_config:
                _log_resolved_config(run_preset, output_path)
            continue

        tree = build_tree_from_preset(run_preset)
        branches_config = load_component_config("branches", run_preset.branches)

        if show_resolved_config:
            _log_resolved_config(run_preset, output_path)

        if validate_only:
            logger.info("Validated runtime configuration for seed={} output={}", run_preset.seed, output_path)
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Generating seed={} tasks={} samples_per_task={} output={}",
            run_preset.seed,
            run_preset.num_tasks,
            run_preset.num_samples_per_task,
            output_path,
        )
        metadata: dict[str, Any] = {
            "schema_version": "2.1",
            "preset_identifier": run_preset.preset_id or "custom",
            "preset_alias": preset_alias(run_preset),
            "dataset_name": output_path.stem,
            "resolved_output_path": str(output_path),
            "command_invocation": command_invocation,
            "resolved_preset_json": run_preset.model_dump(mode="json"),
        }
        generate_and_export_dataset(
            path=str(output_path),
            tree=tree,
            num_updates=run_preset.num_tasks,
            num_samples_per_update=run_preset.num_samples_per_task,
            num_samples_in_first_task=run_preset.num_samples_first_task,
            seed=run_preset.seed,
            generation_metadata=metadata,
            class_names=list(branches_config.classes),
        )
        logger.info("Generated dataset: {}", output_path)

    if skip_existing and skipped:
        logger.info("Skipped {} existing dataset(s).", skipped)


def run_generate_request(
    *,
    preset: str,
    dry_run: bool = False,
    validate_only: bool = False,
    show_resolved_config: bool = False,
    show_capabilities: bool = False,
    allow_unsupported_drifts: bool = False,
    skip_existing: bool = False,
) -> None:
    """🚀 Resolve and execute a generation request for one preset."""
    if dry_run and validate_only:
        raise ValueError("--dry-run and --validate-only are mutually exclusive.")

    resolved_name = resolve_drift_preset_name(preset)
    base_preset = load_drift_preset(resolved_name, validate_runtime_capabilities=False)

    if allow_unsupported_drifts:
        logger.warning(
            "Skipping runtime drift capability validation (--allow-unsupported-drifts). Use this only for exploratory research scenarios.",
        )
    else:
        validate_drift_runtime_capabilities(base_preset.drifts, context="CLI runtime")

    plan = _plan_runs(base_preset)
    _validate_unique_output_paths(plan)

    command_invocation = f"python -m endlesswm generate {resolved_name}"

    if not dry_run and not validate_only:
        _copy_preset_alongside_outputs(base_preset, plan)

    _run_generation_flow(
        plan,
        command_invocation=command_invocation,
        dry_run=dry_run,
        validate_only=validate_only,
        show_resolved_config=show_resolved_config,
        show_capabilities=show_capabilities,
        skip_existing=skip_existing,
    )
