"""Candidate batch selection helpers for LBA."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import ceil
from typing import AbstractSet

from .types import SampleRecord


@dataclass(frozen=True)
class BatchCandidate:
    """A contiguous length-sorted window that can become a dynamic batch."""

    start_index: int
    end_index: int
    total_raw_length: int
    total_padded_length: int
    total_padding_length: int
    padding_ratio: float
    earliest_arrival_id: int

    @property
    def record_count(self) -> int:
        return self.end_index - self.start_index + 1


def find_threshold_candidate(
    records: Sequence[SampleRecord],
    prefix_lengths: Sequence[int],
    *,
    max_padded_length: int,
    max_padding_ratio: float,
    recent_arrival_ids: AbstractSet[int],
) -> BatchCandidate | None:
    """Find the best candidate that satisfies the configured padding threshold."""

    candidates = [
        candidate
        for candidate in iter_batch_candidates(
            records,
            prefix_lengths,
            max_padded_length=max_padded_length,
            max_padding_ratio=max_padding_ratio,
        )
        if candidate.padding_ratio <= max_padding_ratio
        and candidate_contains_recent_record(candidate, records, recent_arrival_ids)
    ]
    if not candidates:
        return None
    return min(candidates, key=threshold_candidate_key)


def find_best_candidate(
    records: Sequence[SampleRecord],
    prefix_lengths: Sequence[int],
    *,
    max_padded_length: int,
    max_padding_ratio: float,
) -> BatchCandidate | None:
    """Find the lowest-padding candidate when no threshold candidate is ready."""

    candidates = list(
        iter_batch_candidates(
            records,
            prefix_lengths,
            max_padded_length=max_padded_length,
            max_padding_ratio=max_padding_ratio,
        )
    )
    if not candidates:
        return None

    multi_record_candidates = [
        candidate for candidate in candidates if candidate.record_count > 1
    ]
    return min(multi_record_candidates or candidates, key=best_candidate_key)


def iter_batch_candidates(
    records: Sequence[SampleRecord],
    prefix_lengths: Sequence[int],
    *,
    max_padded_length: int,
    max_padding_ratio: float,
) -> Iterator[BatchCandidate]:
    """Yield candidate windows ending at each length-sorted record."""

    sorted_lengths = [record.length for record in records]
    for end_index, longest_record in enumerate(records):
        if longest_record.length <= 0:
            continue

        max_record_count = max_padded_length // longest_record.length
        if max_record_count <= 0:
            continue

        widest_start_index = max(0, end_index - max_record_count + 1)
        yield make_batch_candidate(
            records,
            prefix_lengths,
            widest_start_index,
            end_index,
        )

        min_length_for_ratio = ceil(longest_record.length * (1 - max_padding_ratio))
        tight_start_index = bisect_left(
            sorted_lengths,
            min_length_for_ratio,
            widest_start_index,
            end_index + 1,
        )
        if tight_start_index <= end_index and tight_start_index != widest_start_index:
            yield make_batch_candidate(
                records,
                prefix_lengths,
                tight_start_index,
                end_index,
            )


def make_batch_candidate(
    records: Sequence[SampleRecord],
    prefix_lengths: Sequence[int],
    start_index: int,
    end_index: int,
) -> BatchCandidate:
    window_records = records[start_index : end_index + 1]
    total_raw_length = prefix_lengths[end_index + 1] - prefix_lengths[start_index]
    longest_length = records[end_index].length
    record_count = end_index - start_index + 1
    total_padded_length = longest_length * record_count
    total_padding_length = total_padded_length - total_raw_length
    padding_ratio = (
        total_padding_length / total_padded_length if total_padded_length else 0.0
    )
    return BatchCandidate(
        start_index=start_index,
        end_index=end_index,
        total_raw_length=total_raw_length,
        total_padded_length=total_padded_length,
        total_padding_length=total_padding_length,
        padding_ratio=padding_ratio,
        earliest_arrival_id=min(record.arrival_id for record in window_records),
    )


def candidate_contains_recent_record(
    candidate: BatchCandidate,
    records: Sequence[SampleRecord],
    recent_arrival_ids: AbstractSet[int],
) -> bool:
    if not recent_arrival_ids:
        return True
    return any(
        record.arrival_id in recent_arrival_ids
        for record in records[candidate.start_index : candidate.end_index + 1]
    )


def threshold_candidate_key(candidate: BatchCandidate) -> tuple[int, float, int, int]:
    return (
        -candidate.total_padded_length,
        candidate.padding_ratio,
        candidate.total_padding_length,
        candidate.earliest_arrival_id,
    )


def best_candidate_key(candidate: BatchCandidate) -> tuple[float, int, int, int]:
    return (
        candidate.padding_ratio,
        candidate.total_padding_length,
        -candidate.total_padded_length,
        candidate.earliest_arrival_id,
    )
