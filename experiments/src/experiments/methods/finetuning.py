"""Plain finetuning method."""

from dataclasses import dataclass

from . import Method


@dataclass
class Finetuning(Method):
    """Plain task-by-task finetuning."""

    name: str = "finetuning"
