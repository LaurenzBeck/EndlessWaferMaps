"""PyTorch dataset abstractions for EndlessWM HDF5 files."""

import os
from typing import Callable, Iterator, Sequence

import h5py
import pandas as pd
import torch


class H5WM(torch.utils.data.Dataset):
    """Multi-label dataset abstraction over EndlessWM HDF5 files.

    When task=None (default), uses global index for cross-task access.
    When task=int, provides access to a specific task only.
    """

    def __init__(
        self,
        file_path: str,
        task: int | None = None,
        transform: dict[str, Callable] | None = None,
        fields: Sequence[str] = ("sample", "multilabel"),
    ):
        self.file_path = file_path
        self.task_idx = int(task) if task is not None else None
        self.transform = transform or {}
        self.fields = tuple(fields)

        self._file = None
        self._pid = None
        self._gindex = None
        self._length = None
        self._version = None

        self.dtypes = {
            "sample": torch.float32,
            "pattern": torch.float32,
            "background": torch.float32,
            "multilabel": torch.float32,
            "masks": torch.uint8,
            "wafer_mask": torch.uint8,
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_pid"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._file = None
        self._pid = None

    def _ensure_open(self):
        pid = os.getpid()
        if self._file is None or self._pid != pid:
            if self._file is not None:
                try:
                    self._file.close()
                except Exception:
                    pass
            self._file = h5py.File(self.file_path, "r")
            self._pid = pid

            if self.task_idx is None and self._gindex is None:
                self._gindex = self._file["meta"]["global_index"][...]

            if self._version is None:
                self._version = self._file.attrs.get("version", "0.0")

    def _close_file_handle(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    def __del__(self):
        self._close_file_handle()

    def __len__(self):
        if self._length is None:
            with h5py.File(self.file_path, "r") as f:
                if self.task_idx is not None:
                    self._length = int(f[f"tasks/{self.task_idx}/multilabel"].shape[0])
                else:
                    self._length = int(f["meta"]["global_index"].shape[0])
        return self._length

    def _needs_channel_expansion(self, field: str, arr) -> bool:
        if field not in ("sample", "pattern", "background"):
            return False
        return arr.ndim == 2

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        self._ensure_open()

        if self.task_idx is not None:
            g = self._file[f"tasks/{self.task_idx}"]
            sample_idx = idx
        else:
            t_idx, sample_idx = self._gindex[idx]
            t_key = str(int(t_idx))
            g = self._file["tasks"][t_key]

        out = []
        for field in self.fields:
            arr = g[field][sample_idx]
            t = torch.from_numpy(arr).to(self.dtypes[field])

            if self._needs_channel_expansion(field, arr):
                t = t.unsqueeze(0)

            if field in self.transform:
                t = self.transform[field](t)
            out.append(t)
        return tuple(out)

    @property
    def labels(self) -> torch.Tensor:
        self._ensure_open()
        if self.task_idx is not None:
            return torch.from_numpy(self._file["tasks"][str(self.task_idx)]["multilabel"][:]).to(self.dtypes["multilabel"])
        return torch.cat(
            [torch.from_numpy(self._file["tasks"][k]["multilabel"][:]) for k in sorted(self._file["tasks"].keys(), key=int)],
        )

    @property
    def samples(self) -> torch.Tensor:
        if self.task_idx is None:
            raise ValueError("samples property only available in single-task mode (task != None)")

        self._ensure_open()
        arr = self._file["tasks"][str(self.task_idx)]["sample"][:]
        t = torch.from_numpy(arr).to(self.dtypes["sample"])

        if arr.ndim == 3 and arr.shape[1] != 1:
            t = t.unsqueeze(1)

        return t

    @property
    def tasks(self) -> Iterator["H5WM"]:
        self._ensure_open()
        for task_key in sorted(self._file["tasks"].keys(), key=int):
            yield H5WM(self.file_path, int(task_key), self.transform, self.fields)

    @property
    def parameters(self) -> pd.DataFrame:
        self._ensure_open()

        if self.task_idx is not None:
            g = self._file[f"tasks/{self.task_idx}"]

            if "parameters" in g and isinstance(g["parameters"], h5py.Group):
                data = {}
                for key in g["parameters"].keys():
                    data[key] = g["parameters"][key][:]
                return pd.DataFrame(data)

            params = g["parameters"][:]
            return pd.DataFrame({name: params[name] for name in params.dtype.names})

        dfs = []
        for task_key in sorted(self._file["tasks"].keys(), key=int):
            task_idx = int(task_key)
            task_dset = H5WM(self.file_path, task_idx, self.transform, self.fields)
            df = task_dset.parameters
            df.insert(0, "task", task_idx)
            df.insert(1, "sample_idx", range(len(df)))
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)


H5WMTask = H5WM

__all__ = ["H5WM", "H5WMTask"]
