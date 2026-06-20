"""Shared internal types for LBA."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

LengthFn = Callable[[Any], int]
CollateFn = Callable[[list[Any]], Any]


@dataclass(frozen=True)
class LengthRecord:
    """A raw sample with its effective length before global arrival order exists."""

    sample: Any
    length: int
    index: int | None = None


@dataclass(frozen=True)
class SampleRecord:
    """A raw sample plus the metadata LBA needs to plan batches."""

    sample: Any
    length: int
    arrival_id: int
    index: int | None = None


@dataclass(frozen=True)
class BatchPlan:
    """A planned dynamic batch before the original collate function runs."""

    records: Sequence[SampleRecord]
    raw_length_sum: int
    padded_length: int
    padding_length: int
    padding_ratio: float
    reason: str

    @property
    def samples(self) -> list[Any]:
        return [record.sample for record in self.records]
