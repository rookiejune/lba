"""Top-level dataloader adapter."""

from __future__ import annotations

import warnings
from collections.abc import Generator, Iterable, Iterator
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from .config import DEFAULT_PREFETCH_BATCHES, LBAConfig
from .estimator import LengthBudgetResolver
from .logging_utils import create_run_logger
from .metrics import PaddingStats, PlannerStats, padding_ratio_reduction
from .planner import BatchPlanner
from .prefetch import prefetch_iterator
from .source import build_source_loader
from .types import BatchPlan, LengthFn, LengthRecord, SampleRecord


class LengthBatchingAdapter:
    """Wrap a dataloader and prepare length-based dynamic batches."""

    def __init__(
        self,
        dataloader: DataLoader,
        *,
        len_fn: LengthFn,
        max_padded_length: int | None = None,
        warmup_batches: int | None = None,
        max_cache_samples: int = 8192,
        max_padding_ratio: float = 0.05,
        prefetch_batches: int = DEFAULT_PREFETCH_BATCHES,
        spill_dir: str | Path | None = None,
        log_dir: str | Path | None = None,
    ) -> None:
        if len_fn is None:
            raise TypeError("len_fn is required.")

        self.dataloader = dataloader
        self.len_fn = len_fn
        self.original_collate_fn = dataloader.collate_fn
        self.config = LBAConfig(
            max_padded_length=max_padded_length,
            warmup_batches=warmup_batches,
            max_cache_samples=max_cache_samples,
            max_padding_ratio=max_padding_ratio,
            prefetch_batches=prefetch_batches,
            spill_dir=spill_dir,
            log_dir=log_dir,
        )
        self.logger, self.log_path = create_run_logger(log_dir)
        self._active_max_padded_length: int | None = None

        warnings.warn(f"LBA log file: {self.log_path}", stacklevel=2)
        self.logger.info("LBA log file: %s", self.log_path)
        if max_padded_length is not None:
            warnings.warn(
                "max_padded_length is set explicitly and overrides warmup inference.",
                stacklevel=2,
            )
            self.logger.warning("explicit max_padded_length=%s", max_padded_length)

    @property
    def max_padded_length(self) -> int | None:
        return self.config.max_padded_length

    def __iter__(self) -> Iterator[Any]:
        if self._distributed_is_initialized():
            if self.config.prefetch_batches > 0:
                self.logger.info(
                    "disabled LBA prefetch for torch.distributed iteration"
                )
            return self._iter_distributed_sync()

        iterator = self._iter_sync()
        if self.config.prefetch_batches > 0:
            return prefetch_iterator(iterator, self.config.prefetch_batches)
        return iterator

    def _iter_sync(self) -> Generator[Any, None, None]:
        return self._iter_planned(distributed=False)

    def _iter_distributed_sync(self) -> Generator[Any, None, None]:
        return self._iter_planned(distributed=True)

    def _iter_planned(self, *, distributed: bool) -> Generator[Any, None, None]:
        record_loader = build_source_loader(self.dataloader, self.len_fn)
        length_record_iter = iter(record_loader)
        resolver = LengthBudgetResolver(self.config, self.dataloader)
        before_padding_stats = PaddingStats()
        after_padding_stats = PaddingStats()
        warmup_batches = self._collect_warmup_batches(
            length_record_iter,
            resolver,
            before_padding_stats,
            distributed=distributed,
        )
        resolved_max_padded_length = self._resolve_iteration_max_padded_length(
            resolver,
            warmup_batches,
            distributed=distributed,
        )
        self._active_max_padded_length = resolved_max_padded_length
        planner = self._build_planner(resolved_max_padded_length)

        try:
            yield from self._iter_plans(
                planner,
                warmup_batches,
                length_record_iter,
                before_padding_stats,
                after_padding_stats,
                distributed=distributed,
            )
        finally:
            self._log_run_summary(
                before_padding_stats,
                after_padding_stats,
                planner.stats,
            )
            planner.close()

    def _collect_warmup_batches(
        self,
        length_record_iter: Iterator[list[LengthRecord]],
        resolver: LengthBudgetResolver,
        before_padding_stats: PaddingStats,
        *,
        distributed: bool,
    ) -> list[list[LengthRecord]]:
        warmup_batches: list[list[LengthRecord]] = []
        if self.config.max_padded_length is not None:
            return warmup_batches

        for warmup_index in range(resolver.warmup_batch_count()):
            has_batch, length_records = self._next_source_batch(
                length_record_iter,
                distributed=distributed,
                context=f"warmup batch {warmup_index}",
            )
            if not has_batch:
                break
            before_padding_stats.add_length_records(length_records)
            warmup_batches.append(length_records)
        return warmup_batches

    def _resolve_iteration_max_padded_length(
        self,
        resolver: LengthBudgetResolver,
        warmup_batches: Iterable[Iterable[LengthRecord]],
        *,
        distributed: bool,
    ) -> int:
        warmup_length_records = self._flatten_length_records(warmup_batches)
        resolved_max_padded_length = resolver.resolve(warmup_length_records)
        if distributed:
            return self._sync_distributed_max_padded_length(resolved_max_padded_length)
        return resolved_max_padded_length

    def _build_planner(self, max_padded_length: int) -> BatchPlanner:
        return BatchPlanner(
            max_padded_length=max_padded_length,
            max_cache_samples=self.config.max_cache_samples,
            max_padding_ratio=self.config.max_padding_ratio,
            spill_dir=self.config.spill_dir,
            logger=self.logger,
        )

    def _iter_plans(
        self,
        planner: BatchPlanner,
        warmup_batches: Iterable[Iterable[LengthRecord]],
        length_record_iter: Iterator[list[LengthRecord]],
        before_padding_stats: PaddingStats,
        after_padding_stats: PaddingStats,
        *,
        distributed: bool,
    ) -> Generator[Any, None, None]:
        arrival_id = 0
        for length_records in self._iter_length_record_groups(
            warmup_batches,
            length_record_iter,
            before_padding_stats,
            distributed=distributed,
        ):
            sample_records, arrival_id = self._assign_arrival_ids(
                length_records,
                arrival_id,
            )
            plans = self._plans_after_add(
                planner,
                sample_records,
            )
            yield from self._collate_plans(plans, after_padding_stats)

        if distributed:
            yield from self._collate_plans(
                self._distributed_flush_plans(planner),
                after_padding_stats,
            )
        else:
            yield from self._collate_plans(planner.flush(), after_padding_stats)

    def _iter_length_record_groups(
        self,
        warmup_batches: Iterable[Iterable[LengthRecord]],
        length_record_iter: Iterator[list[LengthRecord]],
        before_padding_stats: PaddingStats,
        *,
        distributed: bool,
    ) -> Generator[Iterable[LengthRecord], None, None]:
        if distributed:
            yield from warmup_batches
        else:
            warmup_length_records = self._flatten_length_records(warmup_batches)
            if warmup_length_records:
                yield warmup_length_records

        yield from self._remaining_source_batches(
            length_record_iter,
            before_padding_stats,
            distributed=distributed,
        )

    def _remaining_source_batches(
        self,
        length_record_iter: Iterator[list[LengthRecord]],
        before_padding_stats: PaddingStats,
        *,
        distributed: bool,
    ) -> Generator[list[LengthRecord], None, None]:
        while True:
            has_batch, length_records = self._next_source_batch(
                length_record_iter,
                distributed=distributed,
                context="source batch",
            )
            if not has_batch:
                break
            before_padding_stats.add_length_records(length_records)
            yield length_records

    def _plans_after_add(
        self,
        planner: BatchPlanner,
        sample_records: list[SampleRecord],
    ) -> list[BatchPlan]:
        planner.add_records(sample_records)
        plan = planner.pop_ready()
        if plan is None:
            return []
        return [plan]

    def _collate_plans(
        self, plans: Iterable[BatchPlan], after_padding_stats: PaddingStats
    ) -> Generator[Any, None, None]:
        for plan in plans:
            yield self._collate_recorded_plan(plan, after_padding_stats)

    @staticmethod
    def _flatten_length_records(
        length_record_batches: Iterable[Iterable[LengthRecord]],
    ) -> list[LengthRecord]:
        return [
            record
            for length_records in length_record_batches
            for record in length_records
        ]

    def _assign_arrival_ids(
        self, length_records: Iterable[LengthRecord], next_arrival_id: int
    ) -> tuple[list[SampleRecord], int]:
        sample_records: list[SampleRecord] = []
        for length_record in length_records:
            sample_records.append(
                SampleRecord(
                    sample=length_record.sample,
                    length=length_record.length,
                    arrival_id=next_arrival_id,
                    index=length_record.index,
                )
            )
            next_arrival_id += 1
        return sample_records, next_arrival_id

    def _next_source_batch(
        self,
        length_record_iter: Iterator[list[LengthRecord]],
        *,
        distributed: bool,
        context: str,
    ) -> tuple[bool, list[LengthRecord]]:
        try:
            length_records = list(next(length_record_iter))
            local_has_batch = 1
        except StopIteration:
            length_records = []
            local_has_batch = 0

        if not distributed:
            return bool(local_has_batch), length_records

        min_has_batch, max_has_batch = self._distributed_int_min_max(local_has_batch)
        if min_has_batch != max_has_batch:
            raise RuntimeError(
                "LBA distributed mode requires every rank to receive the same "
                f"number of source DataLoader batches; mismatch during {context}."
            )
        return bool(local_has_batch), length_records

    def _sync_distributed_max_padded_length(self, local_value: int) -> int:
        min_value, max_value = self._distributed_int_min_max(local_value)
        if self.config.max_padded_length is not None and min_value != max_value:
            raise RuntimeError(
                "LBA distributed mode requires identical explicit "
                "max_padded_length on every rank."
            )
        if self.config.max_padded_length is None and local_value != max_value:
            self.logger.info(
                "using distributed max_padded_length=%s from local_value=%s",
                max_value,
                local_value,
            )
        return max_value

    def _distributed_flush_plans(self, planner: BatchPlanner) -> list[BatchPlan]:
        local_records = planner.drain_records()
        if self._records_have_indices(local_records):
            return self._distributed_index_flush_plans(local_records)
        return self._distributed_object_flush_plans(local_records)

    def _distributed_index_flush_plans(
        self, local_records: list[SampleRecord]
    ) -> list[BatchPlan]:
        local_metadata = [
            (self._require_record_index(record), record.length)
            for record in local_records
        ]
        gathered_metadata: list[list[tuple[int, int]]] = [
            [] for _ in range(dist.get_world_size())
        ]
        dist.all_gather_object(gathered_metadata, local_metadata)

        global_records = self._records_from_index_metadata(gathered_metadata)
        assigned_plans = self._assigned_global_flush_plans(global_records)
        return [self._materialize_index_plan(plan) for plan in assigned_plans]

    def _distributed_object_flush_plans(
        self, local_records: list[SampleRecord]
    ) -> list[BatchPlan]:
        gathered_records: list[list[SampleRecord]] = [
            [] for _ in range(dist.get_world_size())
        ]
        dist.all_gather_object(gathered_records, local_records)

        global_records = self._reassign_arrival_ids(gathered_records)
        return self._assigned_global_flush_plans(global_records)

    def _assigned_global_flush_plans(
        self, global_records: list[SampleRecord]
    ) -> list[BatchPlan]:
        global_plans = self._plan_global_flush_records(global_records)
        target_count = self._round_up_to_world_size(len(global_plans))
        global_plans = self._split_plans_to_count(global_plans, target_count)

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return [
            plan
            for index, plan in enumerate(global_plans)
            if index % world_size == rank
        ]

    @staticmethod
    def _records_have_indices(records: Iterable[SampleRecord]) -> bool:
        return all(record.index is not None for record in records)

    @staticmethod
    def _require_record_index(record: SampleRecord) -> int:
        if record.index is None:
            raise RuntimeError("LBA expected a sample index for distributed flush.")
        return record.index

    @staticmethod
    def _records_from_index_metadata(
        gathered_metadata: Iterable[Iterable[tuple[int, int]]],
    ) -> list[SampleRecord]:
        global_records: list[SampleRecord] = []
        for rank_metadata in gathered_metadata:
            for sample_index, length in rank_metadata:
                global_records.append(
                    SampleRecord(
                        sample=sample_index,
                        length=length,
                        arrival_id=len(global_records),
                        index=sample_index,
                    )
                )
        return global_records

    def _materialize_index_plan(self, plan: BatchPlan) -> BatchPlan:
        records = [
            SampleRecord(
                sample=self.dataloader.dataset[self._require_record_index(record)],
                length=record.length,
                arrival_id=record.arrival_id,
                index=record.index,
            )
            for record in plan.records
        ]
        return self._make_batch_plan(records, plan.reason)

    @staticmethod
    def _reassign_arrival_ids(
        gathered_records: Iterable[Iterable[SampleRecord]],
    ) -> list[SampleRecord]:
        global_records: list[SampleRecord] = []
        for rank_records in gathered_records:
            for record in rank_records:
                global_records.append(
                    SampleRecord(
                        sample=record.sample,
                        length=record.length,
                        arrival_id=len(global_records),
                        index=record.index,
                    )
                )
        return global_records

    def _plan_global_flush_records(
        self, records: list[SampleRecord]
    ) -> list[BatchPlan]:
        if not records:
            return []
        max_padded_length = self._require_active_max_padded_length()
        planner = BatchPlanner(
            max_padded_length=max_padded_length,
            max_cache_samples=max(len(records), 1),
            max_padding_ratio=self.config.max_padding_ratio,
            spill_dir=None,
            logger=self.logger,
        )
        try:
            planner.add_records(records, allow_spill=False)
            return list(planner.flush())
        finally:
            planner.close()

    def _round_up_to_world_size(self, value: int) -> int:
        world_size = dist.get_world_size()
        remainder = value % world_size
        if remainder == 0:
            return value
        return value + world_size - remainder

    def _require_active_max_padded_length(self) -> int:
        if self._active_max_padded_length is None:
            raise RuntimeError("LBA has no active max_padded_length for flushing.")
        return self._active_max_padded_length

    def _split_plans_to_count(
        self, plans: list[BatchPlan], target_count: int
    ) -> list[BatchPlan]:
        if target_count < len(plans):
            raise RuntimeError(
                "LBA distributed batch synchronization received an invalid target."
            )
        total_record_count = sum(len(plan.records) for plan in plans)
        if target_count > total_record_count:
            raise RuntimeError(
                "LBA distributed flush could not create enough non-empty batches "
                "for every rank."
            )

        split_plans = list(plans)
        while len(split_plans) < target_count:
            split_index = self._largest_splittable_plan_index(split_plans)
            if split_index is None:
                raise RuntimeError(
                    "LBA distributed mode could not split local batches enough to "
                    "match the number of steps on every rank. Make sure every rank "
                    "uses equally sized source DataLoader batches."
                )

            plan = split_plans.pop(split_index)
            midpoint = len(plan.records) // 2
            left_plan = self._make_batch_plan(plan.records[:midpoint], plan.reason)
            right_plan = self._make_batch_plan(plan.records[midpoint:], plan.reason)
            split_plans[split_index:split_index] = [left_plan, right_plan]

        return split_plans

    @staticmethod
    def _largest_splittable_plan_index(plans: list[BatchPlan]) -> int | None:
        best_index: int | None = None
        best_size = 1
        for index, plan in enumerate(plans):
            plan_size = len(plan.records)
            if plan_size > best_size:
                best_size = plan_size
                best_index = index
        return best_index

    @staticmethod
    def _make_batch_plan(records: Iterable[SampleRecord], reason: str) -> BatchPlan:
        ordered_records = tuple(sorted(records, key=lambda record: record.arrival_id))
        raw_length_sum = sum(record.length for record in ordered_records)
        padded_length = max(record.length for record in ordered_records) * len(
            ordered_records
        )
        padding_length = padded_length - raw_length_sum
        padding_ratio = padding_length / padded_length if padded_length else 0.0
        return BatchPlan(
            records=ordered_records,
            raw_length_sum=raw_length_sum,
            padded_length=padded_length,
            padding_length=padding_length,
            padding_ratio=padding_ratio,
            reason=reason,
        )

    def _collate_recorded_plan(
        self, plan: BatchPlan, after_padding_stats: PaddingStats
    ) -> Any:
        after_padding_stats.add_plan(plan)
        return self._collate_plan(plan)

    def _collate_plan(self, plan: BatchPlan) -> Any:
        if plan.reason == "oversized":
            oversized_sample = plan.records[0].sample
            active_max_padded_length = self._active_max_padded_length
            warnings.warn(
                f"LBA oversized sample length={plan.records[0].length} "
                f"max_padded_length={active_max_padded_length}: {oversized_sample!r}",
                stacklevel=2,
            )
            self.logger.warning(
                "oversized sample length=%s max_padded_length=%s sample=%r",
                plan.records[0].length,
                active_max_padded_length,
                oversized_sample,
            )
        return self.original_collate_fn(plan.samples)

    def _log_run_summary(
        self,
        before_padding_stats: PaddingStats,
        after_padding_stats: PaddingStats,
        planner_stats: PlannerStats,
    ) -> None:
        reduction = padding_ratio_reduction(before_padding_stats, after_padding_stats)
        self._log_summary_section(
            "padding",
            (
                (
                    "before_padding_ratio",
                    self._format_ratio(before_padding_stats.global_padding_ratio),
                ),
                (
                    "before_mean_batch_padding_ratio",
                    self._format_ratio(before_padding_stats.mean_batch_padding_ratio),
                ),
                (
                    "after_padding_ratio",
                    self._format_ratio(after_padding_stats.global_padding_ratio),
                ),
                (
                    "after_mean_batch_padding_ratio",
                    self._format_ratio(after_padding_stats.mean_batch_padding_ratio),
                ),
                ("padding_ratio_reduction", self._format_percent(reduction)),
            ),
        )
        self._log_summary_section(
            "lengths",
            (
                ("before_batches", before_padding_stats.batch_count),
                ("before_samples", before_padding_stats.sample_count),
                ("before_raw_length_sum", before_padding_stats.raw_length_sum),
                ("before_padded_length_sum", before_padding_stats.padded_length_sum),
                ("before_padding_length_sum", before_padding_stats.padding_length_sum),
                ("after_batches", after_padding_stats.batch_count),
                ("after_samples", after_padding_stats.sample_count),
                ("after_raw_length_sum", after_padding_stats.raw_length_sum),
                ("after_padded_length_sum", after_padding_stats.padded_length_sum),
                ("after_padding_length_sum", after_padding_stats.padding_length_sum),
            ),
        )
        self._log_summary_section(
            "planner",
            (
                ("planned_batches", after_padding_stats.planned_batch_count),
                ("oversized_batches", after_padding_stats.oversized_batch_count),
                ("other_batches", after_padding_stats.other_batch_count),
                ("sort_time_seconds", f"{planner_stats.sort_time_seconds:.6f}"),
                ("sort_calls", planner_stats.sort_call_count),
                (
                    "average_sort_time_ms",
                    self._format_milliseconds(planner_stats.average_sort_time_ms),
                ),
                ("records_sorted_total", planner_stats.records_sorted_total),
                ("max_cache_size_seen", planner_stats.max_cache_size_seen),
                ("spill_events", planner_stats.spill_event_count),
                ("spilled_records", planner_stats.spilled_record_count),
            ),
        )

    def _log_summary_section(
        self, section: str, fields: Iterable[tuple[str, object]]
    ) -> None:
        self.logger.info("LBA summary %s %s", section, self._format_fields(fields))

    @staticmethod
    def _format_fields(fields: Iterable[tuple[str, object]]) -> str:
        return " ".join(f"{key}={value}" for key, value in fields)

    @staticmethod
    def _format_ratio(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:.4f}"

    @staticmethod
    def _format_percent(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value * 100:.2f}%"

    @staticmethod
    def _format_milliseconds(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:.3f}"

    @staticmethod
    def _distributed_is_initialized() -> bool:
        return dist.is_available() and dist.is_initialized()

    @staticmethod
    def _distributed_int_min_max(value: int) -> tuple[int, int]:
        device = LengthBatchingAdapter._distributed_tensor_device()
        min_tensor = torch.tensor(value, dtype=torch.long, device=device)
        max_tensor = torch.tensor(value, dtype=torch.long, device=device)
        dist.all_reduce(min_tensor, op=dist.ReduceOp.MIN)
        dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
        return int(min_tensor.item()), int(max_tensor.item())

    @staticmethod
    def _distributed_tensor_device() -> torch.device:
        backend = str(dist.get_backend()).lower()
        if "nccl" not in backend:
            return torch.device("cpu")
        if not torch.cuda.is_available():
            raise RuntimeError("LBA distributed mode with NCCL requires CUDA.")
        return torch.device("cuda", torch.cuda.current_device())


LBA = LengthBatchingAdapter
