"""Source-loader construction for reading length records."""

from __future__ import annotations

import operator
from typing import Any

from torch.utils.data import DataLoader

from .types import LengthFn, LengthRecord


class RecordCollator:
    """Collate raw samples into records with lengths."""

    def __init__(self, len_fn: LengthFn) -> None:
        self.len_fn = len_fn

    def __call__(self, samples: list[Any]) -> list[LengthRecord]:
        length_records: list[LengthRecord] = []
        for sample in samples:
            sample_length = operator.index(self.len_fn(sample))
            if sample_length <= 0:
                raise ValueError("len_fn must return a positive integer.")
            length_records.append(LengthRecord(sample=sample, length=sample_length))
        return length_records


def build_source_loader(dataloader: DataLoader, len_fn: LengthFn) -> DataLoader:
    """Build a loader that yields lists of LengthRecord."""

    collate_fn = RecordCollator(len_fn)
    loader_kwargs: dict[str, Any] = {
        "batch_sampler": dataloader.batch_sampler,
        "num_workers": dataloader.num_workers,
        "collate_fn": collate_fn,
        "pin_memory": dataloader.pin_memory,
        "timeout": dataloader.timeout,
        "worker_init_fn": dataloader.worker_init_fn,
        "persistent_workers": dataloader.persistent_workers,
    }

    if dataloader.multiprocessing_context is not None:
        loader_kwargs["multiprocessing_context"] = dataloader.multiprocessing_context

    if dataloader.generator is not None:
        loader_kwargs["generator"] = dataloader.generator

    if dataloader.num_workers > 0 and dataloader.prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = dataloader.prefetch_factor

    if dataloader.pin_memory_device:
        loader_kwargs["pin_memory_device"] = dataloader.pin_memory_device

    return DataLoader(dataloader.dataset, **loader_kwargs)
