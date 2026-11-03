"""⚙️ Utility functions for reproducibility and label analysis."""

import os
import random

from rich.console import Console
from rich.table import Table
import numpy as np
import torch

cudnn_deterministic = True


def seed_everything(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility.

    Args:
        seed: Random seed to use across all libraries (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic


def _print_section(title: str) -> None:
    print(f"\n{'=' * 114}\n{title}\n{'=' * 114}")


def _print_test_summary(results: dict[str, dict[int, dict[str, float]]], num_tasks: int) -> None:
    if not results:
        return

    _print_section("Test stream summary")
    for stream_name, task_results in results.items():
        metric_names = sorted({name for values in task_results.values() for name in values if name != "loss"})
        metric_names = ["loss", *metric_names]
        table = Table(title=f"stream={stream_name}")
        table.add_column("metric")
        for task_index in range(num_tasks):
            table.add_column(f"task {task_index}", justify="right")

        for metric_name in metric_names:
            row = [metric_name]
            for task_index in range(num_tasks):
                value = task_results.get(task_index, {}).get(metric_name)
                row.append("-" if value is None else f"{value:.4f}")
            table.add_row(*row)
        Console(width=10_000, color_system=None).print(table)
