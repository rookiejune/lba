"""Top-level dataloader adapter."""

from __future__ import annotations

import warnings
from collections.abc import Generator, Iterable, Iterator
from pathlib import Path
from typing import Any

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
        iterator = self._iter_sync()
        if self.config.prefetch_batches > 0:
            return prefetch_iterator(iterator, self.config.prefetch_batches)
        return iterator

    def _iter_sync(self) -> Generator[Any, None, None]:
        record_loader = build_source_loader(self.dataloader, self.len_fn)
        length_record_iter = iter(record_loader)
        resolver = LengthBudgetResolver(self.config, self.dataloader)
        before_padding_stats = PaddingStats()
        after_padding_stats = PaddingStats()
        warmup_length_records: list[LengthRecord] = []

        if self.config.max_padded_length is None:
            for _ in range(resolver.warmup_batch_count()):
                try:
                    length_records = next(length_record_iter)
                except StopIteration:
                    break
                before_padding_stats.add_length_records(length_records)
                warmup_length_records.extend(length_records)

        resolved_max_padded_length = resolver.resolve(warmup_length_records)
        self._active_max_padded_length = resolved_max_padded_length
        planner = BatchPlanner(
            max_padded_length=resolved_max_padded_length,
            max_cache_samples=self.config.max_cache_samples,
            max_padding_ratio=self.config.max_padding_ratio,
            spill_dir=self.config.spill_dir,
            logger=self.logger,
        )
        arrival_id = 0

        try:
            if warmup_length_records:
                sample_records, arrival_id = self._assign_arrival_ids(
                    warmup_length_records,
                    arrival_id,
                )
                planner.add_records(sample_records)
                plan = planner.pop_ready()
                if plan is not None:
                    yield self._collate_recorded_plan(plan, after_padding_stats)

            for length_records in length_record_iter:
                before_padding_stats.add_length_records(length_records)
                sample_records, arrival_id = self._assign_arrival_ids(
                    length_records,
                    arrival_id,
                )
                planner.add_records(sample_records)
                plan = planner.pop_ready()
                if plan is not None:
                    yield self._collate_recorded_plan(plan, after_padding_stats)

            for plan in planner.flush():
                yield self._collate_recorded_plan(plan, after_padding_stats)
        finally:
            self._log_run_summary(
                before_padding_stats,
                after_padding_stats,
                planner.stats,
            )
            planner.close()

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
                )
            )
            next_arrival_id += 1
        return sample_records, next_arrival_id

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
        self.logger.info(
            "LBA summary padding "
            "before_padding_ratio=%s before_mean_batch_padding_ratio=%s "
            "after_padding_ratio=%s after_mean_batch_padding_ratio=%s "
            "padding_ratio_reduction=%s",
            self._format_ratio(before_padding_stats.global_padding_ratio),
            self._format_ratio(before_padding_stats.mean_batch_padding_ratio),
            self._format_ratio(after_padding_stats.global_padding_ratio),
            self._format_ratio(after_padding_stats.mean_batch_padding_ratio),
            self._format_percent(reduction),
        )
        self.logger.info(
            "LBA summary lengths "
            "before_batches=%s before_samples=%s before_raw_length_sum=%s "
            "before_padded_length_sum=%s before_padding_length_sum=%s "
            "after_batches=%s after_samples=%s after_raw_length_sum=%s "
            "after_padded_length_sum=%s after_padding_length_sum=%s",
            before_padding_stats.batch_count,
            before_padding_stats.sample_count,
            before_padding_stats.raw_length_sum,
            before_padding_stats.padded_length_sum,
            before_padding_stats.padding_length_sum,
            after_padding_stats.batch_count,
            after_padding_stats.sample_count,
            after_padding_stats.raw_length_sum,
            after_padding_stats.padded_length_sum,
            after_padding_stats.padding_length_sum,
        )
        self.logger.info(
            "LBA summary planner "
            "planned_batches=%s oversized_batches=%s other_batches=%s "
            "sort_time_seconds=%.6f sort_calls=%s average_sort_time_ms=%s "
            "records_sorted_total=%s max_cache_size_seen=%s "
            "spill_events=%s spilled_records=%s",
            after_padding_stats.planned_batch_count,
            after_padding_stats.oversized_batch_count,
            after_padding_stats.other_batch_count,
            planner_stats.sort_time_seconds,
            planner_stats.sort_call_count,
            self._format_milliseconds(planner_stats.average_sort_time_ms),
            planner_stats.records_sorted_total,
            planner_stats.max_cache_size_seen,
            planner_stats.spill_event_count,
            planner_stats.spilled_record_count,
        )

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


LBA = LengthBatchingAdapter
