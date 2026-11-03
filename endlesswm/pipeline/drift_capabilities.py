"""# 🧭 Drift Strategy Support

Supported strategies for each drift axis.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from endlesswm.pipeline.config.schema import DriftConfig


@dataclass(frozen=True)
class DriftAxisCapability:
    """Supported strategies for one drift axis."""

    axis_label: str
    get_strategy: Callable[[DriftConfig], str]
    supported_strategies: frozenset[str]


_COVARIATE_AND_CONCEPT_STRATEGIES = frozenset(
    {
        "fixed_full",
        "fixed_random_fraction",
        "random_fraction",
        "fixed_center_fraction",
        "anchor_sweep",
        "brownian",
        "teleporting_brownian",
    },
)

RUNTIME_DRIFT_CAPABILITIES: tuple[DriftAxisCapability, ...] = (
    DriftAxisCapability(
        axis_label="Covariate drift",
        get_strategy=lambda drift: drift.covariate.strategy,
        supported_strategies=_COVARIATE_AND_CONCEPT_STRATEGIES,
    ),
    DriftAxisCapability(
        axis_label="Concept drift",
        get_strategy=lambda drift: drift.concept.strategy,
        supported_strategies=_COVARIATE_AND_CONCEPT_STRATEGIES,
    ),
    DriftAxisCapability(
        axis_label="Prior cardinality",
        get_strategy=lambda drift: drift.prior.cardinality_schedule,
        supported_strategies=frozenset({"fixed", "linear_transition", "resampled_subset", "single_label"}),
    ),
    DriftAxisCapability(
        axis_label="Prior class",
        get_strategy=lambda drift: drift.prior.class_schedule,
        supported_strategies=frozenset(
            {
                "fixed",
                "linear_transition",
                "resampled_subset",
                "class_incremental",
                "class_incremental_replay",
            },
        ),
    ),
)


def get_runtime_drift_capability_map() -> dict[str, list[str]]:
    """Return supported strategies per drift axis."""
    return {axis.axis_label: sorted(axis.supported_strategies) for axis in RUNTIME_DRIFT_CAPABILITIES}


def get_drift_capability_report(drift_config: DriftConfig) -> dict[str, dict[str, object]]:
    """Build a support summary per drift axis."""
    report: dict[str, dict[str, object]] = {}
    for axis in RUNTIME_DRIFT_CAPABILITIES:
        selected = axis.get_strategy(drift_config)
        supported = selected in axis.supported_strategies
        report[axis.axis_label] = {
            "selected_strategy": selected,
            "status": "ready" if supported else "experimental",
            "supported_by_runtime": supported,
            "supported_strategies": sorted(axis.supported_strategies),
        }
    return report


def validate_drift_runtime_capabilities(drift_config: DriftConfig, *, context: str) -> None:
    """Check that all selected drift strategies are supported."""
    for axis in RUNTIME_DRIFT_CAPABILITIES:
        selected = axis.get_strategy(drift_config)
        if selected not in axis.supported_strategies:
            available = ", ".join(sorted(axis.supported_strategies))
            raise NotImplementedError(
                f"{axis.axis_label} strategy '{selected}' is not supported in {context}. Available strategies: {available}.",
            )
