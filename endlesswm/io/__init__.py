"""PyTorch dataset classes and HDF5 export APIs."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .datasets import H5WM, H5WMTask
    from .export import generate_and_export_dataset
    from .snr import compute_sample_snr, snr_db, snr_energy, snr_variance

__all__ = [
    "H5WM",
    "H5WMTask",
    "compute_sample_snr",
    "generate_and_export_dataset",
    "snr_db",
    "snr_energy",
    "snr_variance",
]


def __getattr__(name: str) -> Any:
    if name in {"H5WM", "H5WMTask"}:
        module = import_module(".datasets", __name__)
        value = getattr(module, name)
    elif name == "generate_and_export_dataset":
        module = import_module(".export", __name__)
        value = getattr(module, name)
    elif name in {"compute_sample_snr", "snr_db", "snr_energy", "snr_variance"}:
        module = import_module(".snr", __name__)
        value = getattr(module, name)
    else:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
