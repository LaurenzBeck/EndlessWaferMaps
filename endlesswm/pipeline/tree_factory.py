"""# 🏭 Tree Factory

Build SamplingTrees from config objects.
"""

# ruff: noqa: EM101, EM102, PLR0911, PLR0912, PLR0915, PLR2004, TRY003

from copy import deepcopy
from numbers import Real

import numpy as np
from scipy.ndimage import gaussian_filter
from streamgen.nodes import TransformNode
from streamgen.samplers.tree import SamplingTree
from streamgen.transforms import noop

from endlesswm.geometry import linear_gradient, linear_radial_gradient
from endlesswm.noise import beta_weibull, impulse_noise, pink_noise
from endlesswm.patterns import create_pattern_nodes
from endlesswm.pipeline.config.schema import (
    BranchesConfig,
    CardinalityPriorConfig,
    ClassPriorConfig,
    DriftConfig,
    PostprocessingConfig,
    ShapeScheduleConfig,
    StemConfig,
)
from endlesswm.trees import combine_signals, init_output_dictionary, mask_sample
from endlesswm.utils import affine, generate_and_combine, ignore_first_arg, operate_on_key, sample_and_store_params

_RANGE_EPSILON = 1e-9
_SINGLE_LABEL_RESAMPLE_ATTEMPTS = 128
_RANGE_SCHEDULE_STRATEGIES = frozenset(
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


def _is_numeric_range(value: object) -> bool:
    """Return True if value is a 2-element numeric range."""
    return isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[0], Real) and isinstance(value[1], Real)


def _normalize_range(value: list[float] | tuple[float, float]) -> tuple[float, float]:
    """Normalize range order and cast values to float."""
    low = float(value[0])
    high = float(value[1])
    return (low, high) if low <= high else (high, low)


def _normalize_probabilities(values: np.ndarray) -> np.ndarray:
    """Return a non-negative probability vector with unit sum."""
    probs = np.asarray(values, dtype=np.float64)
    probs = np.where(np.isfinite(probs), probs, 0.0)
    probs = np.clip(probs, 0.0, None)

    total = float(np.sum(probs))
    if total <= _RANGE_EPSILON:
        if probs.size == 0:
            raise ValueError("Cannot normalize an empty probability vector.")
        return np.ones(probs.size, dtype=np.float64) / probs.size

    return probs / total


def _compute_mask_cardinality(sample: dict) -> int:
    """Return the number of active class masks in a sampled wafer map."""
    masks = np.asarray(sample.get("masks"), dtype=np.float32)
    if masks.size == 0:
        return 0

    valid_masks = np.nan_to_num(masks, nan=0.0)
    return int(np.count_nonzero(np.any(valid_masks != 0.0, axis=(1, 2))))


class _SingleLabelResamplingTree(SamplingTree):
    """SamplingTree variant that retries until a valid single-label sample is produced."""

    def __init__(self, *args, max_resample_attempts: int = _SINGLE_LABEL_RESAMPLE_ATTEMPTS, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_resample_attempts = max(1, int(max_resample_attempts))

    def sample(self, *args, **kwargs):  # noqa: ANN002, ANN003
        last_cardinality = 0

        for _ in range(self.max_resample_attempts):
            sample = super().sample(*args, **kwargs)
            last_cardinality = _compute_mask_cardinality(sample)
            if last_cardinality == 1:
                return sample

        raise RuntimeError(
            "Single-label sampling failed to produce exactly one active mask after "
            f"{self.max_resample_attempts} attempts; last sampled cardinality was {last_cardinality}."
        )


def _uniform_distribution(size: int, support_indices: np.ndarray) -> np.ndarray:
    """Build a uniform distribution over support_indices."""
    if size <= 0:
        raise ValueError("Distribution size must be positive.")

    support = np.asarray(support_indices, dtype=np.int64)
    if support.size == 0:
        return np.ones(size, dtype=np.float64) / size

    distribution = np.zeros(size, dtype=np.float64)
    distribution[support] = 1.0 / support.size
    return distribution


def _support_from_distribution(distribution: np.ndarray) -> np.ndarray:
    """Return indices with non-zero probability; fallback to all indices."""
    support = np.flatnonzero(np.asarray(distribution, dtype=np.float64) > _RANGE_EPSILON)
    if support.size:
        return support
    return np.arange(len(distribution), dtype=np.int64)


def _as_distribution(candidate: object, *, expected_size: int, label: str) -> np.ndarray:
    """Validate and normalize a user-provided distribution."""
    if not isinstance(candidate, (list, tuple, np.ndarray)):
        raise TypeError(f"{label} must be a list/tuple/array with length {expected_size}.")

    vector = np.asarray(candidate, dtype=np.float64)
    if vector.shape != (expected_size,):
        raise ValueError(f"{label} must have length {expected_size}, got {vector.shape}.")

    return _normalize_probabilities(vector)


def _resolve_prior_axis_params(prior_params: dict, *, axis: str) -> dict:
    """Resolve axis-specific prior params with shared defaults."""
    shared = {
        key: value for key, value in prior_params.items() if key not in {"cardinality", "cardinality_params", "class", "class_params"}
    }

    axis_specific: dict = {}
    for key in (axis, f"{axis}_params"):
        value = prior_params.get(key)
        if isinstance(value, dict):
            axis_specific.update(value)

    return shared | axis_specific


def _compute_transition_probabilities(cardinality_distribution: np.ndarray) -> np.ndarray:
    """Compute P(cardinality=k -> k+1) from cardinality distribution."""
    distribution = _normalize_probabilities(cardinality_distribution)
    transition = np.zeros_like(distribution, dtype=np.float64)

    for idx, value in enumerate(distribution):
        tail_sum = float(np.sum(distribution[idx:]))
        transition[idx] = 1.0 - (value / tail_sum) if tail_sum > _RANGE_EPSILON else 0.0

    return transition


def _build_cardinality_distribution_schedule(
    strategy: str,
    base_distribution: np.ndarray,
    num_tasks: int,
    rng: np.random.Generator,
    strategy_params: dict,
) -> np.ndarray:
    """Build a per-task schedule for cardinality distributions."""
    base = _normalize_probabilities(base_distribution)
    support = _support_from_distribution(base)
    size = base.size

    if strategy == "fixed":
        return np.tile(base, (num_tasks, 1))

    if strategy == "single_label":
        one_hot = np.zeros(size, dtype=np.float64)
        target_idx = 1 if size > 1 else 0
        one_hot[target_idx] = 1.0
        return np.tile(one_hot, (num_tasks, 1))

    if strategy == "linear_transition":
        start_distribution = strategy_params.get("start_distribution")
        end_distribution = strategy_params.get("end_distribution")

        if start_distribution is None:
            start = base
        else:
            start = _as_distribution(start_distribution, expected_size=size, label="cardinality.start_distribution")

        if end_distribution is None:
            end = _uniform_distribution(size, support)
        else:
            end = _as_distribution(end_distribution, expected_size=size, label="cardinality.end_distribution")

        if num_tasks == 1:
            return start.reshape(1, size)

        alphas = np.linspace(0.0, 1.0, num_tasks, dtype=np.float64)
        return np.stack([_normalize_probabilities((1.0 - alpha) * start + alpha * end) for alpha in alphas], axis=0)

    if strategy == "resampled_subset":
        active_count = int(strategy_params.get("active_count", max(1, min(3, support.size))))
        active_count = max(1, min(active_count, support.size))
        resample_every = max(1, int(strategy_params.get("resample_every_tasks", 1)))
        use_base_weights = bool(strategy_params.get("use_base_weights", True))

        schedule: list[np.ndarray] = []
        active: np.ndarray | None = None

        for task_idx in range(num_tasks):
            if task_idx % resample_every == 0 or active is None:
                active = np.asarray(rng.choice(support, size=active_count, replace=False), dtype=np.int64)

            distribution = np.zeros(size, dtype=np.float64)
            weights = base[active] if use_base_weights else np.ones(active_count, dtype=np.float64)
            distribution[active] = _normalize_probabilities(weights)
            schedule.append(distribution)

        return np.stack(schedule, axis=0)

    raise NotImplementedError(
        f"Prior cardinality strategy '{strategy}' is not supported. Supported: fixed, linear_transition, resampled_subset, single_label.",
    )


def _build_class_distribution_schedule(
    strategy: str,
    base_distribution: np.ndarray,
    num_tasks: int,
    rng: np.random.Generator,
    strategy_params: dict,
) -> np.ndarray:
    """Build a per-task schedule for class distributions."""
    base = _normalize_probabilities(base_distribution)
    support = _support_from_distribution(base)
    size = base.size

    if strategy == "fixed":
        return np.tile(base, (num_tasks, 1))

    if strategy == "linear_transition":
        start_distribution = strategy_params.get("start_distribution")
        end_distribution = strategy_params.get("end_distribution")

        if start_distribution is None:
            start = base
        else:
            start = _as_distribution(start_distribution, expected_size=size, label="class.start_distribution")

        if end_distribution is None:
            end = _uniform_distribution(size, support)
        else:
            end = _as_distribution(end_distribution, expected_size=size, label="class.end_distribution")

        if num_tasks == 1:
            return start.reshape(1, size)

        alphas = np.linspace(0.0, 1.0, num_tasks, dtype=np.float64)
        return np.stack([_normalize_probabilities((1.0 - alpha) * start + alpha * end) for alpha in alphas], axis=0)

    if strategy == "resampled_subset":
        active_count = int(strategy_params.get("active_count", max(1, min(3, support.size))))
        active_count = max(1, min(active_count, support.size))
        resample_every = max(1, int(strategy_params.get("resample_every_tasks", 1)))
        use_base_weights = bool(strategy_params.get("use_base_weights", True))

        schedule: list[np.ndarray] = []
        active: np.ndarray | None = None

        for task_idx in range(num_tasks):
            if task_idx % resample_every == 0 or active is None:
                active = np.asarray(rng.choice(support, size=active_count, replace=False), dtype=np.int64)

            distribution = np.zeros(size, dtype=np.float64)
            weights = base[active] if use_base_weights else np.ones(active_count, dtype=np.float64)
            distribution[active] = _normalize_probabilities(weights)
            schedule.append(distribution)

        return np.stack(schedule, axis=0)

    if strategy in {"class_incremental", "class_incremental_replay"}:
        classes_per_step = max(1, int(strategy_params.get("classes_per_step", 1)))
        replay = strategy == "class_incremental_replay"
        shuffle_order = bool(strategy_params.get("shuffle_order", False))

        class_order = np.asarray(support, dtype=np.int64)
        if shuffle_order:
            class_order = np.asarray(rng.permutation(class_order), dtype=np.int64)

        num_stages = max(1, int(np.ceil(class_order.size / classes_per_step)))
        stage_length = max(1, int(strategy_params.get("stage_length", int(np.ceil(num_tasks / num_stages)))))

        schedule: list[np.ndarray] = []
        for task_idx in range(num_tasks):
            stage_idx = min(num_stages - 1, task_idx // stage_length)
            end = min(class_order.size, (stage_idx + 1) * classes_per_step)
            start = 0 if replay else max(0, end - classes_per_step)
            active = class_order[start:end]

            if active.size == 0:
                active = class_order[:1]

            distribution = np.zeros(size, dtype=np.float64)
            distribution[active] = 1.0 / active.size
            schedule.append(distribution)

        return np.stack(schedule, axis=0)

    raise NotImplementedError(
        "Prior class strategy "
        f"'{strategy}' is not supported. Supported: fixed, linear_transition, resampled_subset, "
        "class_incremental, class_incremental_replay.",
    )


def _resolve_subrange_mode(strategy_params: dict) -> str:
    """Resolve sub-range anchor mode used by covariate schedules."""
    mode_raw = str(strategy_params.get("subrange_mode", "random")).lower()
    aliases = {
        "random": "random",
        "start": "start",
        "middle": "middle",
        "center": "middle",
        "end": "end",
        "deterministic_start": "start",
        "deterministic_middle": "middle",
        "deterministic_end": "end",
    }

    if mode_raw not in aliases:
        raise ValueError(
            f"Unknown covariate 'subrange_mode' '{mode_raw}'. "
            "Supported: random, start, middle, end, deterministic_start, deterministic_middle, deterministic_end.",
        )

    return aliases[mode_raw]


def _anchor_index(anchor_mode: str, size: int, rng: np.random.Generator) -> int:
    """Choose an anchor index for deterministic or random-start schedules."""
    if size <= 1 or anchor_mode == "start":
        return 0
    if anchor_mode == "middle":
        return size // 2
    if anchor_mode == "end":
        return size - 1
    return int(rng.integers(0, size))


def _resolve_covariate_scope_roots(
    strategy_params: dict,
    background_roots: set[str],
    pattern_roots: set[str],
) -> set[str] | None:
    """Resolve which top-level parameter groups receive covariate schedules."""
    scope = str(strategy_params.get("signal_scope", "background")).lower()
    if scope == "background":
        selected = set(background_roots)
    elif scope == "pattern":
        selected = set(pattern_roots)
    elif scope == "both":
        selected = set(background_roots) | set(pattern_roots)
    else:
        raise ValueError(f"Unknown covariate signal_scope '{scope}'. Supported: background, pattern, both.")

    target_roots = strategy_params.get("target_roots")
    if target_roots is not None:
        explicit = {target_roots} if isinstance(target_roots, str) else {str(root) for root in target_roots}
        if scope == "both":
            selected = explicit
        else:
            selected &= explicit

    return selected or None


def _cardinality_transition_scope_name(cardinality: int, path_signature: tuple[int, ...]) -> str:
    """Return scope name used for cardinality-transition branch probabilities."""
    path = "_".join(str(index) for index in path_signature) if path_signature else "root"
    return f"cardinality_transition_k{cardinality + 1}_path_{path}"


def _pattern_selection_scope_name(
    cardinality: int,
    available_indices: np.ndarray,
    path_signature: tuple[int, ...],
) -> str:
    """Return scope name used for class-selection branch probabilities."""
    path = "_".join(str(index) for index in path_signature) if path_signature else "root"
    choices = "_".join(str(int(index)) for index in np.asarray(available_indices, dtype=np.int64).tolist())
    return f"pattern_selection_k{cardinality}_path_{path}_choices_{choices}"


def _project_class_distribution_to_choices(
    class_distribution: np.ndarray,
    available_indices: np.ndarray,
) -> list[float]:
    """Project global class distribution onto currently available class choices."""
    available = np.asarray(available_indices, dtype=np.int64)
    return _normalize_probabilities(np.asarray(class_distribution, dtype=np.float64)[available]).tolist()


def _has_class_distribution_mass(class_distribution: np.ndarray, available_indices: np.ndarray) -> bool:
    """Return whether available choices contain positive class-prior mass."""
    available = np.asarray(available_indices, dtype=np.int64)
    if available.size == 0:
        return False
    return float(np.sum(np.asarray(class_distribution, dtype=np.float64)[available])) > _RANGE_EPSILON


def _transition_probs_for_available_classes(
    *,
    cardinality: int,
    possible_patterns: np.ndarray,
    static_transition_probabilities: np.ndarray,
    static_class_distribution: np.ndarray,
    class_schedule: np.ndarray | None,
    transition_schedule_by_cardinality: dict[int, list[list[float]]] | None,
) -> tuple[list[float] | None, dict | None]:
    """Resolve transition probabilities while respecting class-prior support."""
    available = np.asarray(np.where(possible_patterns == 1)[0], dtype=np.int64)

    base_schedule = None
    if transition_schedule_by_cardinality is not None and cardinality in transition_schedule_by_cardinality:
        base_schedule = transition_schedule_by_cardinality[cardinality]

    if class_schedule is not None or base_schedule is not None:
        if base_schedule is None:
            static_transition = static_transition_probabilities[cardinality]
            base_schedule = [
                [
                    float(np.clip(1.0 - task_transition, 0.0, 1.0)),
                    float(np.clip(task_transition, 0.0, 1.0)),
                ]
                for task_transition in np.repeat(static_transition, class_schedule.shape[0])
            ]

        class_distributions = class_schedule
        if class_distributions is None:
            class_distributions = np.repeat(static_class_distribution.reshape(1, -1), len(base_schedule), axis=0)

        schedule = []
        for task_distribution, task_probs in zip(class_distributions, base_schedule, strict=True):
            continue_prob = float(task_probs[1]) if _has_class_distribution_mass(task_distribution, available) else 0.0
            schedule.append(
                [
                    float(np.clip(1.0 - continue_prob, 0.0, 1.0)),
                    float(np.clip(continue_prob, 0.0, 1.0)),
                ],
            )
        return None, {"schedule": schedule, "strategy": "hold"}

    transition_prob = float(np.clip(static_transition_probabilities[cardinality], 0.0, 1.0))
    if not _has_class_distribution_mass(static_class_distribution, available):
        transition_prob = 0.0
    return [1.0 - transition_prob, transition_prob], None


def _construct_cardinality_branches_with_prior(
    *,
    cardinality: int,
    possible_patterns: np.ndarray,
    classes: list[str],
    co_occurrence_matrix: np.ndarray,
    max_cardinality: int,
    static_transition_probabilities: np.ndarray,
    static_class_distribution: np.ndarray,
    class_schedule: np.ndarray | None,
    transition_schedule_by_cardinality: dict[int, list[list[float]]] | None,
    prior_schedule_params: dict,
    path_signature: tuple[int, ...],
) -> dict:
    """Recursively construct cardinality branches with optional scheduled transition probabilities."""
    transition_scope = _cardinality_transition_scope_name(cardinality, path_signature)
    branch: dict = {"name": transition_scope}

    static_probs, scheduled_probs = _transition_probs_for_available_classes(
        cardinality=cardinality,
        possible_patterns=possible_patterns,
        static_transition_probabilities=static_transition_probabilities,
        static_class_distribution=static_class_distribution,
        class_schedule=class_schedule,
        transition_schedule_by_cardinality=transition_schedule_by_cardinality,
    )
    if scheduled_probs is not None:
        prior_schedule_params.setdefault(transition_scope, {})["probs"] = scheduled_probs
    else:
        branch["probs"] = static_probs

    branch[f"cardinality={cardinality}"] = [noop]
    branch[f"cardinality={cardinality + 1}"] = [
        _construct_pattern_branches_with_prior(
            cardinality=cardinality + 1,
            possible_patterns=possible_patterns,
            classes=classes,
            co_occurrence_matrix=co_occurrence_matrix,
            max_cardinality=max_cardinality,
            static_transition_probabilities=static_transition_probabilities,
            static_class_distribution=static_class_distribution,
            class_schedule=class_schedule,
            transition_schedule_by_cardinality=transition_schedule_by_cardinality,
            prior_schedule_params=prior_schedule_params,
            path_signature=path_signature,
        ),
    ]

    return branch


def _construct_pattern_branches_with_prior(
    *,
    cardinality: int,
    possible_patterns: np.ndarray,
    classes: list[str],
    co_occurrence_matrix: np.ndarray,
    max_cardinality: int,
    static_transition_probabilities: np.ndarray,
    static_class_distribution: np.ndarray,
    class_schedule: np.ndarray | None,
    transition_schedule_by_cardinality: dict[int, list[list[float]]] | None,
    prior_schedule_params: dict,
    path_signature: tuple[int, ...],
) -> dict:
    """Recursively construct pattern branches with optional scheduled class probabilities."""
    available = np.asarray(np.where(possible_patterns == 1)[0], dtype=np.int64)
    if available.size == 0:
        return {
            "name": f"pattern_selection_k{cardinality}_fallback",
            "probs": [1.0],
            "no_pattern": [noop],
        }

    selection_scope = _pattern_selection_scope_name(cardinality, available, path_signature)
    branch: dict = {"name": selection_scope}

    if class_schedule is not None:
        scope_params = prior_schedule_params.setdefault(selection_scope, {})
        if "probs" not in scope_params:
            scope_params["probs"] = {
                "schedule": [_project_class_distribution_to_choices(task_distribution, available) for task_distribution in class_schedule],
                "strategy": "hold",
            }
    else:
        branch["probs"] = _project_class_distribution_to_choices(static_class_distribution, available)

    for class_idx in available:
        class_name = classes[int(class_idx)]
        class_path = (*path_signature, int(class_idx))
        class_nodes = [
            *create_pattern_nodes(class_name, idx=int(class_idx)),
        ]

        if cardinality < max_cardinality:
            class_nodes.append(
                _construct_cardinality_branches_with_prior(
                    cardinality=cardinality,
                    possible_patterns=possible_patterns & co_occurrence_matrix[class_idx, :],
                    classes=classes,
                    co_occurrence_matrix=co_occurrence_matrix,
                    max_cardinality=max_cardinality,
                    static_transition_probabilities=static_transition_probabilities,
                    static_class_distribution=static_class_distribution,
                    class_schedule=class_schedule,
                    transition_schedule_by_cardinality=transition_schedule_by_cardinality,
                    prior_schedule_params=prior_schedule_params,
                    path_signature=class_path,
                ),
            )

        branch[class_name] = class_nodes

    return branch


def _sample_fraction_range(
    full_range: tuple[float, float],
    fraction: float,
    rng: np.random.Generator,
    *,
    anchor_mode: str = "random",
) -> tuple[float, float]:
    """Sample a sub-range with width=fraction*full_range width."""
    low, high = full_range
    full_width = max(high - low, _RANGE_EPSILON)
    fraction = float(np.clip(fraction, _RANGE_EPSILON, 1.0))
    width = max(full_width * fraction, _RANGE_EPSILON)

    if width >= full_width - _RANGE_EPSILON:
        return (low, high)

    if anchor_mode == "start":
        start = low
    elif anchor_mode == "middle":
        start = low + 0.5 * (full_width - width)
    elif anchor_mode == "end":
        start = high - width
    else:
        start = float(rng.uniform(low, high - width))

    return (start, start + width)


def _create_range_schedule(
    base_range: tuple[float, float],
    strategy: str,
    strategy_params: dict,
    num_tasks: int,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    """Create a task-wise schedule for one numeric range parameter."""
    full_low, full_high = _normalize_range(base_range)
    full_range = (full_low, full_high)
    if abs(full_high - full_low) <= _RANGE_EPSILON:
        return [full_range for _ in range(num_tasks)]

    full_width = max(full_high - full_low, _RANGE_EPSILON)
    width_fraction = float(strategy_params.get("fraction", strategy_params.get("range_fraction", 0.35)))
    anchor_mode = _resolve_subrange_mode(strategy_params)

    def _normalized_to_range(normalized_low: float) -> tuple[float, float]:
        normalized_low = float(np.clip(normalized_low, 0.0, 1.0))
        clamped_fraction = float(np.clip(width_fraction, _RANGE_EPSILON, 1.0))
        feasible_width = (1.0 - clamped_fraction) * full_width
        absolute_low = full_low + normalized_low * max(feasible_width, 0.0)
        absolute_high = min(full_high, absolute_low + clamped_fraction * full_width)
        return (float(absolute_low), float(absolute_high))

    def _reflect_unit_interval(value: float) -> float:
        return float(1.0 - abs(1.0 - (value % 2.0)))

    if strategy == "fixed_full":
        return [full_range for _ in range(num_tasks)]

    if strategy == "fixed_random_fraction":
        narrowed = _sample_fraction_range(full_range, width_fraction, rng, anchor_mode="random")
        return [narrowed for _ in range(num_tasks)]

    if strategy == "random_fraction":
        return [_sample_fraction_range(full_range, width_fraction, rng, anchor_mode="random") for _ in range(num_tasks)]

    if strategy == "fixed_center_fraction":
        centered = _sample_fraction_range(full_range, width_fraction, rng, anchor_mode="middle")
        return [centered for _ in range(num_tasks)]

    if strategy == "anchor_sweep":
        if num_tasks <= 1:
            start_idx = _anchor_index(anchor_mode, 2, rng)
            normalized_start = 1.0 if start_idx == 1 else 0.0
            return [_normalized_to_range(normalized_start)]

        direction = int(rng.choice([-1, 1]))
        grid = np.linspace(0.0, 1.0, num=num_tasks, dtype=np.float64)
        normalized_path = grid if direction == 1 else (1.0 - grid)
        return [_normalized_to_range(float(value)) for value in normalized_path]

    if strategy == "brownian":
        sigma = float(np.clip(strategy_params.get("sigma", 0.08), 0.0, 10.0))
        normalized_position = float(rng.uniform(0.0, 1.0))
        schedule = [_normalized_to_range(normalized_position)]

        for _ in range(1, num_tasks):
            normalized_position = _reflect_unit_interval(normalized_position + sigma * float(rng.normal(0.0, 1.0)))
            schedule.append(_normalized_to_range(normalized_position))

        return schedule

    if strategy == "teleporting_brownian":
        sigma = float(np.clip(strategy_params.get("sigma", 0.08), 0.0, 10.0))
        teleport_interval_k = max(1, int(strategy_params.get("k", 4)))

        normalized_position = float(rng.uniform(0.0, 1.0))
        schedule = [_normalized_to_range(normalized_position)]

        for task_idx in range(1, num_tasks):
            if task_idx % (teleport_interval_k + 1) == 0:
                normalized_position = float(rng.uniform(0.0, 1.0))
            else:
                normalized_position = _reflect_unit_interval(normalized_position + sigma * float(rng.normal(0.0, 1.0)))
            schedule.append(_normalized_to_range(normalized_position))

        return schedule

    supported = ", ".join(sorted(_RANGE_SCHEDULE_STRATEGIES))
    raise NotImplementedError(
        f"Covariate drift strategy '{strategy}' is not supported. Supported: {supported}.",
    )


def _apply_covariate_schedules_recursive(
    params: dict,
    strategy: str,
    strategy_params: dict,
    num_tasks: int,
    rng: np.random.Generator,
    *,
    selected_roots: set[str] | None,
    current_root: str | None = None,
) -> None:
    """Apply covariate schedules in-place to all *_r parameters."""
    for key, value in params.items():
        root = key if current_root is None else current_root

        if isinstance(value, dict):
            if "schedule" in value:
                continue
            _apply_covariate_schedules_recursive(
                value,
                strategy,
                strategy_params,
                num_tasks,
                rng,
                selected_roots=selected_roots,
                current_root=root,
            )
            continue

        if key.endswith("_r") and _is_numeric_range(value):
            if selected_roots is not None and root not in selected_roots:
                continue
            schedule = _create_range_schedule(value, strategy, strategy_params, num_tasks, rng)
            params[key] = {"schedule": schedule, "strategy": "hold"}


def create_background_node_registry() -> dict:
    """Create registry of background transformation nodes."""
    return {
        "init_output_dictionary": lambda: TransformNode(init_output_dictionary, emoji="📖"),
        "beta_weibull": lambda: TransformNode(
            sample_and_store_params(operate_on_key(ignore_first_arg(beta_weibull), key="background")),
            emoji="🔔",
        ),
        "pink_noise": lambda: TransformNode(
            sample_and_store_params(operate_on_key(generate_and_combine(pink_noise), key="background")),
            emoji="🪷",
        ),
        "gaussian_filter": lambda: TransformNode(
            sample_and_store_params(operate_on_key(gaussian_filter, key="background")),
            emoji="☁️",
        ),
        "impulse_noise": lambda: TransformNode(
            sample_and_store_params(operate_on_key(generate_and_combine(impulse_noise), key="background")),
            emoji="🔨",
        ),
        "linear_gradient": lambda: TransformNode(
            sample_and_store_params(operate_on_key(generate_and_combine(linear_gradient, "*"), key="background")),
            emoji="📏",
        ),
        "linear_radial_gradient": lambda: TransformNode(
            sample_and_store_params(operate_on_key(generate_and_combine(linear_radial_gradient, "*"), key="background")),
            emoji="🛞",
        ),
        "affine": lambda: TransformNode(
            sample_and_store_params(operate_on_key(affine, key="background")),
            emoji="📐",
        ),
    }


class TreeFactory:
    """Factory for creating SamplingTrees from configurations."""

    @staticmethod
    def create_stem_nodes(config: StemConfig) -> list[TransformNode]:
        """Create stem transformation nodes from config."""
        registry = create_background_node_registry()

        nodes = []
        for node_name in config.nodes:
            if node_name not in registry:
                available = ", ".join(registry.keys())
                raise ValueError(f"Unknown stem node '{node_name}'. Available nodes: {available}")
            nodes.append(registry[node_name]())

        return nodes

    @staticmethod
    def create_pattern_branches(
        config: BranchesConfig,
        drift_config: DriftConfig,
        num_tasks: int,
        rng: np.random.Generator,
        cardinality_prior_config: CardinalityPriorConfig,
        class_prior_config: ClassPriorConfig,
    ) -> tuple[dict, dict]:
        """Create pattern branching structure and branch-probability schedules from config."""
        if config.mode == "single_label":
            base_cardinality_distribution = np.array([0.0, 1.0], dtype=np.float64)
        else:
            base_cardinality_distribution = _as_distribution(
                cardinality_prior_config.distribution,
                expected_size=len(cardinality_prior_config.distribution),
                label="cardinality_prior.distribution",
            )

        num_classes = len(config.classes)
        if class_prior_config.probabilities is not None:
            base_class_distribution = _as_distribution(
                class_prior_config.probabilities,
                expected_size=num_classes,
                label="class_prior.probabilities",
            )
        else:
            base_class_distribution = np.ones(num_classes, dtype=np.float64) / num_classes

        if config.mode == "multi_label" and config.co_occurrence is not None:
            co_occurrence_matrix = np.asarray(config.co_occurrence, dtype=np.int32)
        else:
            co_occurrence_matrix = np.ones((num_classes, num_classes), dtype=np.int32) - np.eye(num_classes, dtype=np.int32)

        prior_params = drift_config.prior.params if isinstance(drift_config.prior.params, dict) else {}
        cardinality_params = _resolve_prior_axis_params(prior_params, axis="cardinality")
        class_params = _resolve_prior_axis_params(prior_params, axis="class")

        cardinality_schedule = _build_cardinality_distribution_schedule(
            strategy=drift_config.prior.cardinality_schedule,
            base_distribution=base_cardinality_distribution,
            num_tasks=num_tasks,
            rng=rng,
            strategy_params=cardinality_params,
        )
        class_schedule = _build_class_distribution_schedule(
            strategy=drift_config.prior.class_schedule,
            base_distribution=base_class_distribution,
            num_tasks=num_tasks,
            rng=rng,
            strategy_params=class_params,
        )

        active_cardinality = np.where(np.any(cardinality_schedule > _RANGE_EPSILON, axis=0))[0]
        max_cardinality = 1 if config.mode == "single_label" or active_cardinality.size == 0 else max(1, int(active_cardinality.max()))
        max_cardinality = min(max_cardinality, num_classes)

        transition_schedule = np.stack(
            [_compute_transition_probabilities(cardinality_distribution) for cardinality_distribution in cardinality_schedule],
            axis=0,
        )

        transition_dynamic = num_tasks > 1 and not np.allclose(transition_schedule, transition_schedule[0], atol=1e-12)
        class_dynamic = num_tasks > 1 and not np.allclose(class_schedule, class_schedule[0], atol=1e-12)

        transition_schedule_by_cardinality: dict[int, list[list[float]]] | None = None
        if transition_dynamic:
            transition_schedule_by_cardinality = {}
            for cardinality in range(max_cardinality):
                transition_schedule_by_cardinality[cardinality] = [
                    [
                        float(np.clip(1.0 - task_transition[cardinality], 0.0, 1.0)),
                        float(np.clip(task_transition[cardinality], 0.0, 1.0)),
                    ]
                    for task_transition in transition_schedule
                ]

        prior_schedule_params: dict = {}
        branches = _construct_cardinality_branches_with_prior(
            cardinality=0,
            possible_patterns=np.ones((num_classes,), dtype=np.int32),
            classes=config.classes,
            co_occurrence_matrix=co_occurrence_matrix,
            max_cardinality=max_cardinality,
            static_transition_probabilities=transition_schedule[0],
            static_class_distribution=class_schedule[0],
            class_schedule=class_schedule if class_dynamic else None,
            transition_schedule_by_cardinality=transition_schedule_by_cardinality,
            prior_schedule_params=prior_schedule_params,
            path_signature=(),
        )

        return branches, prior_schedule_params

    @staticmethod
    def create_postprocessing_node(config: PostprocessingConfig) -> TransformNode | None:
        """Create postprocessing node if needed."""
        if config.mode == "continuous":
            return None

        if config.mode == "binary":
            threshold = config.params.get("threshold", 0.5)
            low_value = config.params.get("low_value", 0)
            high_value = config.params.get("high_value", 1)

            def binary_threshold(
                sample: dict,
                threshold: float = threshold,
                low_value: float = low_value,
                high_value: float = high_value,
            ) -> dict:
                arr = sample["sample"]
                nan_mask = np.isnan(arr)
                # Use np.where on a NaN-free copy so the comparison is well-defined,
                # then restore NaN (wafer mask outside the wafer).
                filled = np.where(nan_mask, 0.0, arr)
                binarized = np.where(filled > threshold, high_value, low_value).astype(np.float32)
                binarized[nan_mask] = np.nan
                sample["sample"] = binarized
                return sample

            return TransformNode(binary_threshold, name="binary_threshold", emoji="🔲")

        if config.mode == "normalized":
            scheme = config.params.get("normalization_scheme", "standard")

            def normalize(sample: dict, scheme: str = scheme) -> dict:
                if scheme == "standard":
                    mean = np.nanmean(sample["sample"])
                    std = np.nanstd(sample["sample"])
                    if std > 0:
                        sample["sample"] = (sample["sample"] - mean) / std
                elif scheme == "minmax":
                    min_val = np.nanmin(sample["sample"])
                    max_val = np.nanmax(sample["sample"])
                    if max_val > min_val:
                        sample["sample"] = (sample["sample"] - min_val) / (max_val - min_val)
                return sample

            return TransformNode(normalize, name="normalize", emoji="📊")

        raise ValueError(f"Unknown postprocessing mode '{config.mode}'. Available modes: continuous, binary, normalized")

    @staticmethod
    def create_shape_schedule(
        config: ShapeScheduleConfig,
        num_tasks: int,
        rng: np.random.Generator,
    ) -> dict:
        """Create shape parameter schedule based on strategy."""
        if config.strategy == "fixed":
            height = config.params.get("height", config.params["width"])
            width = config.params["width"]
            fixed_shape = (height, width)
            return {
                "size": fixed_shape,
                "shape": fixed_shape,
            }

        if config.strategy == "random_range":
            width_range = config.params.get("width_range", [40, 120])
            aspect_ratio_range = config.params.get("aspect_ratio", [0.75, 1.15])
            order = str(config.params.get("order", "random")).lower()

            if order not in {"random", "ascending", "descending"}:
                raise ValueError(f"Unknown random_range shape order '{order}'. Available orders: random, ascending, descending")

            shapes = []
            for _ in range(num_tasks):
                width = int(rng.integers(width_range[0], width_range[1] + 1))
                if rng.random() < 0.8:
                    aspect = rng.uniform(aspect_ratio_range[0], aspect_ratio_range[1])
                    shapes.append((width, int(aspect * width)))
                else:
                    shapes.append((width, width))

            if order == "ascending":
                shapes = sorted(shapes, key=lambda shape: shape[0] * shape[1])
            elif order == "descending":
                shapes = sorted(shapes, key=lambda shape: shape[0] * shape[1], reverse=True)

            return {
                "size": {"schedule": shapes},
                "shape": {"schedule": shapes},
            }

        if config.strategy == "scheduled":
            shapes = [tuple(shape) for shape in config.params["schedule"]]
            if len(shapes) != num_tasks:
                raise ValueError(f"Shape schedule has {len(shapes)} entries but num_tasks={num_tasks}")

            return {
                "size": {"schedule": shapes},
                "shape": {"schedule": shapes},
            }

        if config.strategy == "expanding":
            start_width = config.params.get("start_width", 40)
            end_width = config.params.get("end_width", 120)
            aspect = config.params.get("aspect_ratio", 1.0)

            widths = np.linspace(start_width, end_width, num_tasks, dtype=int)
            shapes = [(int(width), int(width * aspect)) for width in widths]

            return {
                "size": {"schedule": shapes},
                "shape": {"schedule": shapes},
            }

        raise ValueError(f"Unknown shape strategy '{config.strategy}'. Available strategies: random_range, fixed, scheduled, expanding")

    @staticmethod
    def apply_drift_schedules(
        params: dict,
        drift_config: DriftConfig,
        num_tasks: int,
        rng: np.random.Generator,
        *,
        background_roots: set[str],
        pattern_roots: set[str],
    ) -> dict:
        """Apply drift schedules to parameters."""
        concept_strategy = drift_config.concept.strategy
        if concept_strategy not in _RANGE_SCHEDULE_STRATEGIES:
            supported = ", ".join(sorted(_RANGE_SCHEDULE_STRATEGIES))
            raise NotImplementedError(
                f"Concept drift strategy '{concept_strategy}' is not yet implemented in TreeFactory. Available strategies: {supported}.",
            )

        if concept_strategy != "fixed_full":
            concept_params = drift_config.concept.params if isinstance(drift_config.concept.params, dict) else {}

            explicit_scope = concept_params.get("signal_scope")
            if explicit_scope is not None and str(explicit_scope).lower() != "pattern":
                raise ValueError(
                    "Concept drift supports only signal_scope='pattern'. Use covariate drift for background or both-scopes schedules.",
                )

            selected_pattern_roots: set[str] = set(pattern_roots)
            target_roots = concept_params.get("target_roots")
            if target_roots is not None:
                explicit_targets = {str(target_roots)} if isinstance(target_roots, str) else {str(root) for root in target_roots}
                selected_pattern_roots &= explicit_targets
                if not selected_pattern_roots:
                    raise ValueError(
                        "Concept drift 'target_roots' did not match any pattern parameter roots.",
                    )

            _apply_covariate_schedules_recursive(
                params=params,
                strategy=concept_strategy,
                strategy_params=concept_params,
                num_tasks=num_tasks,
                rng=rng,
                selected_roots=selected_pattern_roots,
            )

        covariate_strategy = drift_config.covariate.strategy

        if covariate_strategy != "fixed_full":
            selected_roots = _resolve_covariate_scope_roots(
                drift_config.covariate.params,
                background_roots=background_roots,
                pattern_roots=pattern_roots,
            )
            _apply_covariate_schedules_recursive(
                params=params,
                strategy=covariate_strategy,
                strategy_params=drift_config.covariate.params,
                num_tasks=num_tasks,
                rng=rng,
                selected_roots=selected_roots,
            )

        return params

    @staticmethod
    def from_configs(
        stem_config: StemConfig,
        branches_config: BranchesConfig,
        cardinality_prior_config: CardinalityPriorConfig,
        class_prior_config: ClassPriorConfig,
        postprocessing_config: PostprocessingConfig,
        shape_config: ShapeScheduleConfig,
        drift_config: DriftConfig,
        num_tasks: int,
        seed: int,
    ) -> SamplingTree:
        """Create complete SamplingTree from configuration objects."""
        rng = np.random.default_rng(seed)

        stem_params = deepcopy(stem_config.params)
        branches_params = deepcopy(branches_config.params)

        params = deepcopy(stem_params)
        params.update(deepcopy(branches_params))
        params["num_classes"] = len(branches_config.classes)

        shape_schedule = TreeFactory.create_shape_schedule(shape_config, num_tasks, rng)
        params.update(shape_schedule)

        params = TreeFactory.apply_drift_schedules(
            params,
            drift_config,
            num_tasks,
            rng,
            background_roots=set(stem_params.keys()),
            pattern_roots=set(branches_params.keys()),
        )

        pattern_branches, prior_schedule_params = TreeFactory.create_pattern_branches(
            config=branches_config,
            drift_config=drift_config,
            num_tasks=num_tasks,
            rng=rng,
            cardinality_prior_config=cardinality_prior_config,
            class_prior_config=class_prior_config,
        )
        params.update(prior_schedule_params)

        nodes = [
            *TreeFactory.create_stem_nodes(stem_config),
            pattern_branches,
            TransformNode(combine_signals, name="finalize_sample", emoji="📦"),
        ]

        postproc_node = TreeFactory.create_postprocessing_node(postprocessing_config)
        if postproc_node:
            nodes.append(postproc_node)

        nodes.append(TransformNode(mask_sample, name="apply_wafer_mask", emoji="🌐"))

        tree_class = _SingleLabelResamplingTree if branches_config.mode == "single_label" else SamplingTree
        return tree_class(nodes, params, rng=rng)
