"""# ⚙️ Configuration Schema

Pydantic models for dataset generation settings.
"""

# ruff: noqa: D106

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class StemConfig(BaseModel):
    """Configuration for stem generation nodes."""

    name: str = Field(..., description="Name of the stem configuration")
    description: str = Field("", description="Human-readable description")
    nodes: list[str] = Field(
        ...,
        description="List of stem transformation nodes in order",
        examples=[
            [
                "init_output_dictionary",
                "beta_weibull",
                "pink_noise",
                "gaussian_filter",
                "impulse_noise",
                "linear_gradient",
                "linear_radial_gradient",
                "affine",
            ],
        ],
    )
    params: dict = Field(default_factory=dict, description="Parameters for stem transforms")

    class Config:
        extra = "forbid"


class BranchesConfig(BaseModel):
    """Configuration for branch generation."""

    name: str = Field(..., description="Name of the branches configuration")
    description: str = Field("", description="Human-readable description")
    mode: Literal["single_label", "multi_label"] = Field(
        ...,
        description="Pattern labeling mode: single_label (cardinality=1) or multi_label (cardinality>=1)",
    )
    classes: list[str] = Field(
        default=["scratch", "center_spot", "edge_spot", "center_donut", "border_donut", "edge_ring", "crescent", "reticle"],
        description="List of pattern classes to include",
    )
    co_occurrence: list[list[int]] | None = Field(
        None,
        description="Co-occurrence matrix defining which patterns can appear together in multi_label mode.",
    )
    params: dict = Field(default_factory=dict, description="Parameters for branch generation")

    class Config:
        extra = "forbid"

    @model_validator(mode="after")
    def _validate_classes_and_co_occurrence(self) -> "BranchesConfig":
        from endlesswm.patterns import pattern_generators

        if len(self.classes) != len(set(self.classes)):
            raise ValueError(f"'classes' must be unique, got: {self.classes}")

        unknown = [name for name in self.classes if name not in pattern_generators]
        if unknown:
            available = sorted(pattern_generators.keys())
            raise ValueError(f"Unknown pattern class names in 'classes': {unknown}. Available patterns: {available}")

        if self.co_occurrence is not None:
            expected = len(self.classes)
            rows = len(self.co_occurrence)
            if rows != expected or any(len(row) != expected for row in self.co_occurrence):
                raise ValueError(
                    f"'co_occurrence' must be a square {expected}x{expected} matrix matching len(classes); got shape ({rows}, ...).",
                )

        return self


class PostprocessingConfig(BaseModel):
    """Configuration for post-processing generated samples."""

    name: str = Field(..., description="Name of the postprocessing configuration")
    description: str = Field("", description="Human-readable description")
    mode: Literal["continuous", "binary", "normalized"] = Field(
        ...,
        description="Postprocessing mode (continuous, binary, or normalized)",
    )
    params: dict = Field(
        default_factory=dict,
        description="Mode-specific parameters (e.g., threshold for binary, normalization_scheme for normalized)",
    )

    class Config:
        extra = "forbid"


class ShapeScheduleConfig(BaseModel):
    """Configuration for shape scheduling across tasks."""

    name: str = Field(..., description="Name of the shape schedule configuration")
    description: str = Field("", description="Human-readable description")
    strategy: Literal["random_range", "fixed", "scheduled", "expanding"] = Field(
        ...,
        description="Shape scheduling strategy across tasks",
    )
    params: dict = Field(
        default_factory=dict,
        description="Strategy-specific parameters (e.g., width_range, aspect_ratio, order, schedule)",
    )

    class Config:
        extra = "forbid"


class StreamSizeConfig(BaseModel):
    """Configuration for stream-wide task and sample counts."""

    name: str = Field(..., description="Name of the stream size configuration")
    description: str = Field("", description="Human-readable description")
    num_tasks: int = Field(..., gt=0, description="Number of tasks/updates")
    num_samples_per_task: int = Field(..., gt=0, description="Samples per task")
    num_samples_first_task: int | None = Field(
        None,
        gt=0,
        description="Samples in first task (defaults to num_samples_per_task)",
    )
    num_validation_samples: int | None = Field(
        None,
        gt=0,
        description="Default number of validation samples per task for paired validation datasets.",
    )

    class Config:
        extra = "forbid"


class CardinalityPriorConfig(BaseModel):
    """Base cardinality prior configuration."""

    name: str = Field(..., description="Name of the cardinality prior configuration")
    description: str = Field("", description="Human-readable description")
    distribution: list[float] = Field(..., description="Base cardinality distribution vector")

    class Config:
        extra = "forbid"


class ClassPriorConfig(BaseModel):
    """Base class prior configuration."""

    name: str = Field(..., description="Name of the class prior configuration")
    description: str = Field("", description="Human-readable description")
    probabilities: list[float] | None = Field(
        None,
        description="Base class probabilities. If null, uniform class distribution is used.",
    )

    class Config:
        extra = "forbid"


DriftStrategyLit = Literal[
    "fixed_full",
    "fixed_random_fraction",
    "random_fraction",
    "fixed_center_fraction",
    "anchor_sweep",
    "brownian",
    "teleporting_brownian",
]


class _RangeDriftConfig(BaseModel):
    strategy: DriftStrategyLit = Field(..., description="Drift strategy")
    params: dict = Field(default_factory=dict, description="Strategy-specific parameters")

    @model_validator(mode="before")
    @classmethod
    def _normalize_shortcut_form(cls, value: object) -> object:
        if isinstance(value, str):
            return {"strategy": value, "params": {}}
        if not isinstance(value, dict):
            return value

        data = dict(value)
        params_raw = data.get("params") or {}
        if not isinstance(params_raw, dict):
            raise TypeError("params must be a mapping/dict")

        flat_params = {k: v for k, v in data.items() if k not in {"strategy", "params"}}
        return {"strategy": data.get("strategy"), "params": params_raw | flat_params}

    class Config:
        extra = "forbid"


class CovariateDriftConfig(_RangeDriftConfig):
    """Configuration for covariate drift (background parameter schedules)."""


class ConceptDriftConfig(_RangeDriftConfig):
    """Configuration for concept drift (pattern parameter schedules)."""


class PriorDriftConfig(BaseModel):
    """Configuration for prior drift schedules over base prior distributions."""

    cardinality_schedule: Literal["fixed", "linear_transition", "resampled_subset", "single_label"] = Field(
        "fixed",
        description="Cardinality distribution schedule across tasks",
    )
    class_schedule: Literal[
        "fixed",
        "linear_transition",
        "resampled_subset",
        "class_incremental",
        "class_incremental_replay",
    ] = Field(
        "fixed",
        description="Class distribution schedule across tasks",
    )
    params: dict = Field(default_factory=dict, description="Strategy-specific parameters")

    @model_validator(mode="before")
    @classmethod
    def _normalize_shortcut_form(cls, value: object) -> object:
        if isinstance(value, str):
            return {"cardinality_schedule": value, "class_schedule": value, "params": {}}
        if not isinstance(value, dict):
            return value

        data = dict(value)
        params_raw = data.get("params") or {}
        if not isinstance(params_raw, dict):
            raise TypeError("prior.params must be a mapping/dict")

        flat_params = {k: v for k, v in data.items() if k not in {"cardinality_schedule", "class_schedule", "params"}}
        return {
            "cardinality_schedule": data.get("cardinality_schedule", "fixed"),
            "class_schedule": data.get("class_schedule", "fixed"),
            "params": params_raw | flat_params,
        }

    class Config:
        extra = "forbid"


class DriftConfig(BaseModel):
    """Complete drift configuration."""

    covariate: CovariateDriftConfig = Field(
        default_factory=lambda: CovariateDriftConfig(strategy="fixed_full"),
        description="Covariate drift configuration",
    )
    concept: ConceptDriftConfig = Field(
        default_factory=lambda: ConceptDriftConfig(strategy="fixed_full"),
        description="Concept drift configuration",
    )
    prior: PriorDriftConfig = Field(
        default_factory=PriorDriftConfig,
        description="Prior drift configuration",
    )

    class Config:
        extra = "forbid"


class PresetSweepsConfig(BaseModel):
    """Optional comma-separated sweep values attached to a preset."""

    seed: str | None = Field(None, description="Comma-separated seed sweep values, e.g. '1,2,3'")
    tasks: str | None = Field(None, description="Comma-separated task-count sweep values, e.g. '21,17'")
    samples: str | None = Field(None, description="Comma-separated samples-per-task sweep values, e.g. '200,500'")
    fraction: str | None = Field(None, description="Comma-separated paired fraction sweep values, e.g. '0.1,0.2,0.3'")
    sigma: str | None = Field(None, description="Comma-separated paired sigma sweep values, e.g. '0.04,0.08,0.12'")
    stream_size: str | None = Field(
        None,
        description="Comma-separated paired stream size sweep values, e.g. 'small,medium,large'",
    )

    class Config:
        extra = "forbid"


class DatasetPreset(BaseModel):
    """Complete dataset generation configuration preset."""

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_name_key(cls, data: object) -> object:
        if isinstance(data, dict) and "preset_id" not in data and "name" in data:
            normalized = dict(data)
            normalized["preset_id"] = normalized.pop("name")
            return normalized
        return data

    preset_id: str | None = Field(
        None,
        description="Preset path identifier (for example 'static/iii'). The last path component is the preset alias.",
    )
    output_root: str | None = Field(None, description="Optional output directory override for generated dataset files.")
    description: str = Field("", description="Human-readable description of dataset purpose")

    num_tasks: int = Field(..., gt=0, description="Number of tasks/updates")
    num_samples_per_task: int = Field(..., gt=0, description="Samples per task")
    num_samples_first_task: int | None = Field(
        None,
        gt=0,
        description="Samples in first task (defaults to num_samples_per_task)",
    )

    stem: str = Field(..., description="Stem config name (from configs/stem/)")
    branches: str = Field(..., description="Branches config name (from configs/branches/)")
    cardinality_prior: str = Field(
        "mild_long_tailed",
        description="Cardinality prior config name (from configs/cardinality_priors/)",
    )
    class_prior: str = Field(
        "uniform",
        description="Class prior config name (from configs/class_priors/)",
    )
    postprocessing: str = Field(
        "continuous",
        description="Postprocessing config name (from configs/postprocessing/)",
    )
    shapes: str = Field(
        "random_range",
        description="Shape schedule config name (from configs/shapes/)",
    )

    drifts: DriftConfig = Field(default_factory=DriftConfig, description="Drift configuration")
    seed: int = Field(..., description="Random seed for reproducibility")

    stream_size: str | None = Field(
        None,
        description="Optional stream size component name used to resolve default task and sample counts.",
    )
    num_validation_samples: int | None = Field(
        None,
        gt=0,
        description="Number of validation samples per task for the paired validation dataset.",
    )
    sweeps: PresetSweepsConfig | None = Field(
        None,
        description="Resolved sweep definitions used during CLI generation.",
        exclude=True,
    )

    class Config:
        extra = "forbid"

    @property
    def total_samples(self) -> int:
        """Calculate total number of samples."""
        first_task = self.num_samples_first_task or self.num_samples_per_task
        return first_task + (self.num_tasks - 1) * self.num_samples_per_task
