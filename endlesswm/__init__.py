"""# ♾️🌊🗺️ EndlessWM

A variety of very long, synthetically generated wafermap streams.
"""

from enum import Enum
from importlib import import_module
from typing import TYPE_CHECKING, Any, Literal, TypeAlias


class CombinationStrategy(str, Enum):
    """❓ strategy which defines how to combine two values."""

    ADD = "+"
    """Add the values together."""
    MUL = "*"
    """Multiply the values."""


CombinationStrategyLit: TypeAlias = Literal[
    "+",
    "*",
]

if TYPE_CHECKING:
    from .io.datasets import H5WM, H5WMTask
    from .io.export import generate_and_export_dataset
    from .patterns import classes, co_occurrence, num_classes, pattern2idx

__all__ = [
    "CombinationStrategy",
    "CombinationStrategyLit",
    "H5WM",
    "H5WMTask",
    "classes",
    "co_occurrence",
    "generate_and_export_dataset",
    "num_classes",
    "pattern2idx",
]


def __getattr__(name: str) -> Any:
    if name in {"H5WM", "H5WMTask"}:
        module = import_module(".io.datasets", __name__)
        value = getattr(module, name)
    elif name == "generate_and_export_dataset":
        module = import_module(".io.export", __name__)
        value = getattr(module, name)
    elif name in {"classes", "co_occurrence", "num_classes", "pattern2idx"}:
        module = import_module(".patterns", __name__)
        value = getattr(module, name)
    else:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
