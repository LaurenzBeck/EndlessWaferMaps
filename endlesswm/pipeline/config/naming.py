"""# 🏷️ Dataset Naming

Build dataset output filenames and paths from a preset.

A preset lives at ``endlesswm/configs/drift_presets/<group>/<alias>.yaml``.
The output filename is ``<alias>`` followed by underscore-separated tokens
for each swept axis:

* ``fraction`` -> ``f<value>``
* ``sigma`` -> ``sig<value>``
* ``num_tasks`` -> ``tas<value>``
* ``num_samples_per_task`` -> ``spt<value>``
* ``stream_size`` -> ``<value>``
* ``seed`` -> ``s<value>``

Default output path is ``experiments/data/generated/<group>/<filename>.hdf5``
unless ``output_root`` is set on the preset.
"""

# ruff: noqa: ANN401

from pathlib import Path
from typing import Any

from .schema import DatasetPreset

DATASET_OUTPUT_ROOT = Path("experiments/data/generated")

_AXIS_ORDER: tuple[str, ...] = (
    "fraction",
    "sigma",
    "tasks",
    "samples",
    "stream_size",
    "seed",
)
_AXIS_PREFIX: dict[str, str] = {
    "fraction": "f",
    "sigma": "sig",
    "tasks": "tas",
    "samples": "spt",
    "stream_size": "",
    "seed": "s",
}


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _axis_value(preset: DatasetPreset, axis: str) -> Any:
    match axis:
        case "fraction" | "sigma":
            val = preset.drifts.covariate.params.get(axis)
            return val if val is not None else preset.drifts.concept.params.get(axis)
        case "tasks":
            return preset.num_tasks
        case "samples":
            return preset.num_samples_per_task
        case "stream_size":
            return preset.stream_size
        case "seed":
            return preset.seed
        case _:
            raise ValueError(f"Unknown axis: {axis!r}")


def preset_alias(preset: DatasetPreset) -> str:
    """Return the preset alias (filename stem) from its identifier."""
    return Path(preset.preset_id or "dataset").name


def preset_group(preset: DatasetPreset) -> str | None:
    """Return the group folder for the preset, or ``None`` if ungrouped."""
    if not preset.preset_id:
        return None
    parts = Path(preset.preset_id).parts
    return "/".join(parts[:-1]) if len(parts) > 1 else None


def build_dataset_filename(preset: DatasetPreset) -> str:
    """Build the dataset filename stem (without directory or extension)."""
    tokens = [preset_alias(preset)]
    swept = {axis for axis in _AXIS_ORDER if getattr(preset.sweeps, axis, None) is not None} if preset.sweeps else set()
    for axis in _AXIS_ORDER:
        if axis in swept and (value := _axis_value(preset, axis)) is not None:
            tokens.append(f"{_AXIS_PREFIX[axis]}{_format_value(value)}")
    return "_".join(tokens)


def resolve_output_path(preset: DatasetPreset) -> Path:
    """Resolve the full output path for a preset."""
    filename = f"{build_dataset_filename(preset)}.hdf5"
    if preset.output_root is not None:
        return Path(preset.output_root) / filename
    group = preset_group(preset)
    return DATASET_OUTPUT_ROOT / group / filename if group else DATASET_OUTPUT_ROOT / filename
