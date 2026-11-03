"""# 📂 Configuration Loader

Load YAML config files used to build dataset presets and trees.
"""

# ruff: noqa: EM102, I001, PTH123, TRY003

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ValidationError

from .schema import (
    BranchesConfig,
    CardinalityPriorConfig,
    ClassPriorConfig,
    DatasetPreset,
    PostprocessingConfig,
    ShapeScheduleConfig,
    StemConfig,
    StreamSizeConfig,
)
from .sweeps import parse_float_sweep, parse_int_sweep, parse_string_sweep
from endlesswm.pipeline.drift_capabilities import validate_drift_runtime_capabilities

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
DRIFT_PRESETS_DIR = CONFIG_ROOT / "drift_presets"

ComponentKind = Literal[
    "stem",
    "branches",
    "cardinality_priors",
    "class_priors",
    "postprocessing",
    "shapes",
    "stream_size",
]

_COMPONENT_MODELS: dict[ComponentKind, tuple[type[BaseModel], str]] = {
    "stem": (StemConfig, "Stem"),
    "branches": (BranchesConfig, "Branches"),
    "cardinality_priors": (CardinalityPriorConfig, "Cardinality prior"),
    "class_priors": (ClassPriorConfig, "Class prior"),
    "postprocessing": (PostprocessingConfig, "Postprocessing"),
    "shapes": (ShapeScheduleConfig, "Shape"),
    "stream_size": (StreamSizeConfig, "Stream size"),
}


def list_available_components(kind: ComponentKind) -> list[str]:
    """List all available YAML configurations for a component category."""
    folder = CONFIG_ROOT / kind
    return sorted(p.stem for p in folder.glob("*.yaml")) if folder.exists() else []


def list_all_drift_presets() -> list[str]:
    """Return all drift preset names as relative POSIX paths without suffix."""
    if not DRIFT_PRESETS_DIR.exists():
        return []
    return sorted(str(p.relative_to(DRIFT_PRESETS_DIR).with_suffix("")).replace("\\", "/") for p in DRIFT_PRESETS_DIR.rglob("*.yaml"))


def resolve_drift_preset_name(identifier: str) -> str:
    """Resolve a preset identifier by exact relative path or unique basename."""
    available = list_all_drift_presets()
    if identifier in available:
        return identifier

    matches = [name for name in available if Path(name).name == identifier]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Preset identifier '{identifier}' is ambiguous. Matches: {', '.join(matches)}")

    raise FileNotFoundError(f"Drift preset '{identifier}' not found.")


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Load a YAML file and return it as a dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, encoding="utf-8") as handle:
        try:
            return yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise yaml.YAMLError(f"Invalid YAML in {path}: {exc}") from exc


def load_component_config(kind: ComponentKind, name: str) -> Any:
    """Load and validate a component configuration by category and name."""
    model_cls, label = _COMPONENT_MODELS[kind]
    path = CONFIG_ROOT / kind / f"{name}.yaml"
    try:
        return model_cls(**load_yaml_file(path))
    except FileNotFoundError as exc:
        available = ", ".join(list_available_components(kind))
        raise FileNotFoundError(f"{label} config '{name}' not found. Available: {available}") from exc
    except ValidationError as exc:
        raise ValidationError(f"Invalid {label.lower()} config '{name}': {exc}") from exc


def _as_inline_sweep(value: object) -> str | None:
    """Return comma-separated sweep string when a value encodes multiple entries."""
    if isinstance(value, str):
        token = value.strip()
        return token if token and ("," in token or ":" in token) else None
    if isinstance(value, (list, tuple)):
        tokens = [str(item).strip() for item in value if str(item).strip()]
        return ",".join(tokens) if len(tokens) > 1 else None
    return None


def _inject_shared_drift_params(data: dict[str, Any], shared_params: dict[str, Any]) -> dict[str, Any]:
    """Broadcast top-level drift parameters (fraction, sigma, k) to covariate and concept configs."""
    drifts = data.get("drifts")
    if not shared_params or not isinstance(drifts, dict):
        return data

    normalized_drifts = dict(drifts)
    for drift_key in ("covariate", "concept"):
        raw = normalized_drifts.get(drift_key)
        if raw is None:
            continue
        mapping = {"strategy": raw, "params": {}} if isinstance(raw, str) else dict(raw) if isinstance(raw, dict) else None
        if mapping is None or mapping.get("strategy") == "fixed_full":
            if mapping is not None:
                normalized_drifts[drift_key] = mapping
            continue

        params_raw = mapping.get("params") or {}
        if not isinstance(params_raw, dict):
            raise TypeError(f"{drift_key}.params must be a mapping/dict")

        params = dict(params_raw)
        for k, v in mapping.items():
            if k not in {"strategy", "params"}:
                params.setdefault(k, v)
        for k, v in shared_params.items():
            params.setdefault(k, v)

        normalized_drifts[drift_key] = {"strategy": mapping.get("strategy"), "params": params}

    data["drifts"] = normalized_drifts
    return data


def _normalize_inline_preset_shorthand(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize shorthand YAML fields (priors, inline sweeps, shared drift params)."""
    normalized = dict(data)

    if (priors_raw := normalized.pop("priors", None)) is not None:
        if not isinstance(priors_raw, dict):
            raise TypeError("priors must be a mapping/dict")
        if (card := priors_raw.get("cardinality")) is not None:
            normalized.setdefault("cardinality_prior", str(card).strip())
        if (cls_prior := priors_raw.get("class")) is not None:
            normalized.setdefault("class_prior", str(cls_prior).strip())

    sweeps_raw = normalized.get("sweeps")
    sweeps: dict[str, Any] = dict(sweeps_raw) if isinstance(sweeps_raw, dict) else {}

    for field, sweep_key in (("seed", "seed"), ("num_tasks", "tasks"), ("num_samples_per_task", "samples")):
        if (sweep_str := _as_inline_sweep(normalized.get(field))) is not None:
            sweeps[sweep_key] = sweep_str
            normalized[field] = parse_int_sweep(sweep_str, field)[0]

    if (stream_size_sweep := _as_inline_sweep(normalized.get("stream_size"))) is not None:
        sweeps["stream_size"] = stream_size_sweep
        normalized["stream_size"] = parse_string_sweep(stream_size_sweep, "stream_size")[0]

    shared_drift_params: dict[str, Any] = {}
    for key, parser in (("fraction", parse_float_sweep), ("sigma", parse_float_sweep), ("k", parse_int_sweep)):
        if key not in normalized:
            continue
        raw_val = normalized.pop(key)
        if (sweep_val := _as_inline_sweep(raw_val)) is not None:
            if key in {"fraction", "sigma"}:
                sweeps[key] = sweep_val
            shared_drift_params[key] = parser(sweep_val, key)[0]
        else:
            shared_drift_params[key] = parser(str(raw_val).strip(), key)[0]

    normalized = _inject_shared_drift_params(normalized, shared_drift_params)
    if sweeps:
        normalized["sweeps"] = sweeps
    return normalized


def _apply_stream_size_defaults(data: dict[str, Any]) -> dict[str, Any]:
    """Fill missing task and sample counts from the selected stream size."""
    stream_size_name = data.get("stream_size")
    if not isinstance(stream_size_name, str) or not stream_size_name:
        return data

    size_cfg = load_component_config("stream_size", stream_size_name)
    data.setdefault("num_tasks", size_cfg.num_tasks)
    data.setdefault("num_samples_per_task", size_cfg.num_samples_per_task)
    if size_cfg.num_samples_first_task is not None:
        data.setdefault("num_samples_first_task", size_cfg.num_samples_first_task)
    if "num_validation_samples" not in data:
        data["num_validation_samples"] = size_cfg.num_validation_samples
    return data


def load_drift_preset(name: str, *, validate_runtime_capabilities: bool = True) -> DatasetPreset:
    """Load drift preset configuration by name."""
    try:
        resolved_name = resolve_drift_preset_name(name)
    except FileNotFoundError as exc:
        available = list_all_drift_presets()
        raise FileNotFoundError(f"Drift preset '{name}' not found. Available presets: {', '.join(available)}") from exc

    path = DRIFT_PRESETS_DIR / f"{resolved_name}.yaml"
    data = _normalize_inline_preset_shorthand(load_yaml_file(path))
    data.pop("name", None)
    data["preset_id"] = resolved_name
    data = _apply_stream_size_defaults(data)

    preset = DatasetPreset(**data)
    if validate_runtime_capabilities:
        validate_drift_runtime_capabilities(preset.drifts, context="configuration loader")
    return preset
