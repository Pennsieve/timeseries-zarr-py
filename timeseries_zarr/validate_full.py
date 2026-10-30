"""Recompute stored statistics from the data they summarize.

These checks implement section 8.4 of docs/spec.md. Each reads its arrays in
blocks and returns findings instead of raising. NaN equals NaN and -0.0 equals
+0.0 in every comparison, and the checks of mean, var, and waveform_mean allow
the tolerance of section 8.4.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from timeseries_zarr.runs import check_outside_runs
from timeseries_zarr.types import Finding
from timeseries_zarr.zarr_io import ZarrArray

_BLOCK_ELEMENTS = 2**22
_TOLERANCE = 2.0**-20
_FLOAT32_MAXIMUM = float(np.finfo(np.float32).max)

type _Floats = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LevelArrays:
    """The arrays of one continuous level.

    Attributes:
        path: The path of the level group.
        width: The bin width in samples.
        count: The count array.
        minmax: The minmax array.
        mean: The mean array, or None when the level has none.
        var: The var array, or None when the level has none.
    """

    path: str
    width: int
    count: ZarrArray
    minmax: ZarrArray
    mean: ZarrArray | None
    var: ZarrArray | None


@dataclass(slots=True)
class _Mismatch:
    """The bins of one member that differ from the recomputed values."""

    bins: int = 0
    first: int = 0
    stored: object = None
    expected: object = None

    def add(
        self,
        first_row: int,
        bad: npt.NDArray[np.bool_],
        stored: npt.NDArray[Any],
        expected: npt.NDArray[Any],
    ) -> None:
        """Record the rows where bad is set, for rows from first_row on."""
        positions = np.flatnonzero(bad)
        if positions.size == 0:
            return
        if self.bins == 0:
            position = int(positions[0])
            self.first = first_row + position
            self.stored = stored[position].tolist()
            self.expected = expected[position].tolist()
        self.bins += int(positions.size)


@dataclass(slots=True)
class _Statistics:
    """The statistics of the finite samples of each bin of a block."""

    count: npt.NDArray[np.int64]
    minmax: npt.NDArray[np.float32]
    mean: _Floats
    var: _Floats
    max_abs: _Floats


@dataclass(slots=True)
class _LevelAudit:
    """The mismatches found so far in the members of one level."""

    members: dict[str, _Mismatch] = field(default_factory=dict)

    def record(
        self,
        member: str,
        first_row: int,
        bad: npt.NDArray[np.bool_],
        stored: npt.NDArray[Any],
        expected: npt.NDArray[Any],
    ) -> None:
        """Add the bad rows of one block of member."""
        self.members.setdefault(member, _Mismatch()).add(
            first_row, bad, stored, expected
        )


def _rows_statistics(rows: npt.NDArray[np.float32]) -> _Statistics:
    """Return the statistics of the finite samples in each row of rows."""
    finite = np.isfinite(rows)
    count = finite.sum(axis=1)
    minimum = np.where(finite, rows, np.float32(np.inf)).min(axis=1)
    maximum = np.where(finite, rows, np.float32(-np.inf)).max(axis=1)
    values = np.where(finite, rows, np.float32(0)).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = values.sum(axis=1) / count
        deviation = np.where(finite, values - mean[:, None], 0.0)
        variance = (deviation**2).sum(axis=1) / count
    empty = count == 0
    minimum[empty] = np.nan
    maximum[empty] = np.nan
    minmax = np.stack([minimum, maximum], axis=1)
    max_abs = np.abs(minmax.astype(np.float64)).max(axis=1)
    return _Statistics(count, minmax, mean, variance, max_abs)


def _bin_statistics(block: npt.NDArray[np.float32], width: int) -> _Statistics:
    """Return the statistics of each bin of width samples of block.

    The last bin is short when width does not divide the length of block.
    """
    whole = block.shape[0] - block.shape[0] % width
    parts: list[_Statistics] = []
    if whole:
        parts.append(_rows_statistics(block[:whole].reshape(-1, width)))
    if whole < block.shape[0]:
        parts.append(_rows_statistics(block[whole:].reshape(1, -1)))
    return _Statistics(
        count=np.concatenate([part.count for part in parts]),
        minmax=np.concatenate([part.minmax for part in parts]),
        mean=np.concatenate([part.mean for part in parts]),
        var=np.concatenate([part.var for part in parts]),
        max_abs=np.concatenate([part.max_abs for part in parts]),
    )


def _within_tolerance(
    stored: npt.NDArray[Any],
    reference: _Floats,
    max_abs: _Floats,
    *,
    variance: bool,
) -> npt.NDArray[np.bool_]:
    """Return where the stored values pass the tolerance of section 8.4.

    Args:
        stored: The stored values, in any floating dtype.
        reference: The values recomputed in binary64.
        max_abs: The largest absolute finite sample of each value.
        variance: Whether the values are variances. Otherwise they are
            means.
    """
    values = stored.astype(np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        if variance:
            bound = _TOLERANCE * np.maximum(reference, _TOLERANCE * max_abs**2)
        else:
            bound = _TOLERANCE * np.maximum(np.abs(reference), max_abs)
        close = np.abs(values - reference) <= bound
        too_large = reference > _FLOAT32_MAXIMUM
    return np.where(
        np.isnan(reference),
        np.isnan(values),
        np.where(too_large, values == np.inf, close),
    )


def _exact_mismatch(
    stored: npt.NDArray[Any], expected: npt.NDArray[Any]
) -> npt.NDArray[np.bool_]:
    """Return the rows where stored differs from expected, NaN equal to NaN."""
    differs = ~((stored == expected) | (np.isnan(stored) & np.isnan(expected)))
    if differs.ndim > 1:
        return np.asarray(differs.reshape(differs.shape[0], -1).any(axis=1))
    return np.asarray(differs)


def _compare_block(
    audit: _LevelAudit,
    level: LevelArrays,
    first_bin: int,
    statistics: _Statistics,
) -> None:
    """Compare the members of level with the statistics of one block."""
    rows = statistics.count.shape[0]
    window = slice(first_bin, first_bin + rows)
    stored_count = np.asarray(level.count[window]).astype(np.int64)
    audit.record(
        "count",
        first_bin,
        stored_count != statistics.count,
        stored_count,
        statistics.count,
    )
    stored_minmax = np.asarray(level.minmax[window])
    audit.record(
        "minmax",
        first_bin,
        _exact_mismatch(stored_minmax, statistics.minmax),
        stored_minmax,
        statistics.minmax,
    )
    for name, member, reference in (
        ("mean", level.mean, statistics.mean),
        ("var", level.var, statistics.var),
    ):
        if member is None:
            continue
        stored = np.asarray(member[window])
        passes = _within_tolerance(
            stored, reference, statistics.max_abs, variance=name == "var"
        )
        audit.record(name, first_bin, ~passes, stored, reference)


_RECOMPUTED = {
    "count": "the count",
    "minmax": "the minimum and maximum",
    "mean": "the mean",
    "var": "the population variance",
}


def _audit_findings(level: LevelArrays, audit: _LevelAudit) -> list[Finding]:
    """Return one finding for each member of a level that has mismatches."""
    findings = []
    for name, mismatch in audit.members.items():
        if mismatch.bins:
            findings.append(
                Finding(
                    f"{level.path}/{name}",
                    "8.4",
                    f"bin {mismatch.first} holds {mismatch.stored}, but "
                    f"{_RECOMPUTED[name]} of its finite samples is "
                    f"{mismatch.expected}; {mismatch.bins} bins differ",
                )
            )
    return findings


def check_continuous_levels(
    raw: ZarrArray, levels: Sequence[LevelArrays]
) -> list[Finding]:
    """Check the members of the levels of a channel against its raw samples.

    Reads raw once, in blocks that hold whole bins of every level. The count
    and minmax members must equal the recomputed values. The mean and var
    members must lie within the tolerance of section 8.4 of the values
    recomputed in binary64.

    Args:
        raw: The raw array of the channel.
        levels: The levels to check. Each member must have
            ceil(len(raw) / width) rows.

    Returns:
        One finding for each member that differs, naming its first bin.
    """
    if not levels:
        return []
    unit = math.lcm(*(level.width for level in levels))
    step = unit * max(1, _BLOCK_ELEMENTS // unit)
    audits = [_LevelAudit() for _ in levels]
    for start in range(0, raw.shape[0], step):
        block = np.asarray(raw[start : start + step])
        for level, audit in zip(levels, audits, strict=True):
            statistics = _bin_statistics(block, level.width)
            _compare_block(audit, level, start // level.width, statistics)
    return [
        finding
        for level, audit in zip(levels, audits, strict=True)
        for finding in _audit_findings(level, audit)
    ]


def check_raw_outside_runs(
    path: str, raw: ZarrArray, table: npt.NDArray[np.int64]
) -> list[Finding]:
    """Check that every raw sample outside a run is NaN (section 4.2).

    Args:
        path: The path of the raw array, for the finding.
        raw: The raw array.
        table: The runs table of the channel, which satisfies section 4.2.

    Returns:
        A finding for the first sample that is not NaN, or none.
    """
    for start in range(0, raw.shape[0], _BLOCK_ELEMENTS):
        block = np.asarray(raw[start : start + _BLOCK_ELEMENTS])
        try:
            check_outside_runs(path, block, start, table)
        except ValueError as error:
            message = str(error).removeprefix(f"{path}: ")
            return [Finding(path, "4.2", message)]
    return []


def check_count_level(
    path: str,
    events: npt.NDArray[np.int64],
    labels: npt.NDArray[np.uint16] | None,
    label_count: int,
    offset_us: int,
    period_us: int,
    count: ZarrArray,
) -> list[Finding]:
    """Check a count level against the events it counts (section 5.7).

    Args:
        path: The path of the count array, for the finding.
        events: The event times, at least one.
        labels: The label of each event, or None when every label is 0.
        label_count: The number of labels, the width of count.
        offset_us: The time of bin 0.
        period_us: The width of a bin in microseconds.
        count: The count array, with the shape section 5.7 defines.

    Returns:
        A finding for the first bin that differs, or none.
    """
    bins = (int(events[-1]) - offset_us) // period_us + 1
    if count.shape != (bins, label_count):
        return []
    stored = np.asarray(count[:]).astype(np.int64)
    rows = (events - offset_us) // period_us
    columns = np.zeros(events.shape, np.int64) if labels is None else labels
    counted = (rows >= 0) & (rows < bins) & (columns < label_count)
    flat = rows[counted] * label_count + columns[counted].astype(np.int64)
    expected = np.bincount(flat, minlength=bins * label_count).reshape(
        bins, label_count
    )
    bad = np.argwhere(stored != expected)
    if bad.size == 0:
        return []
    row, column = (int(part) for part in bad[0])
    return [
        Finding(
            path,
            "8.4",
            f"bin {row} holds {int(stored[row, column])} events of label "
            f"{column} but {int(expected[row, column])} events fall in it; "
            f"{len(bad)} counts differ",
        )
    ]


def _label_statistics(
    waveforms: ZarrArray,
    labels: npt.NDArray[np.uint16],
    label_count: int,
) -> tuple[_Floats, npt.NDArray[np.int64], _Floats]:
    """Return the sum, the count, and the largest magnitude of each sample.

    Each result has one row for each label and one column for each sample of
    a waveform, over the finite values of the waveforms of that label.
    """
    width = waveforms.shape[1]
    sums = np.zeros((label_count, width))
    counts = np.zeros((label_count, width), np.int64)
    max_abs = np.zeros((label_count, width))
    step = max(1, _BLOCK_ELEMENTS // width)
    for start in range(0, waveforms.shape[0], step):
        block = np.asarray(waveforms[start : start + step])
        block_labels = labels[start : start + step]
        finite = np.isfinite(block)
        values = np.where(finite, block, np.float32(0)).astype(np.float64)
        for label in np.unique(block_labels):
            if label >= label_count:
                continue
            chosen = block_labels == label
            sums[label] += values[chosen].sum(axis=0)
            counts[label] += finite[chosen].sum(axis=0)
            max_abs[label] = np.maximum(
                max_abs[label], np.abs(values[chosen]).max(axis=0)
            )
    return sums, counts, max_abs


def check_waveform_mean(
    path: str,
    waveforms: ZarrArray,
    waveform_mean: ZarrArray,
    labels: npt.NDArray[np.uint16],
    label_count: int,
) -> list[Finding]:
    """Check waveform_mean against the waveforms it averages (section 5.6).

    Args:
        path: The path of the waveform_mean array, for the finding.
        waveforms: The waveforms array.
        waveform_mean: The waveform_mean array, of shape
            (label_count, len(waveforms[0])).
        labels: The label of each event.
        label_count: The number of labels.

    Returns:
        A finding for the first value that fails the tolerance, or none.
    """
    sums, counts, max_abs = _label_statistics(waveforms, labels, label_count)
    with np.errstate(invalid="ignore", divide="ignore"):
        reference = sums / counts
    stored = np.asarray(waveform_mean[:])
    bad = np.argwhere(
        ~_within_tolerance(stored, reference, max_abs, variance=False)
    )
    if bad.size == 0:
        return []
    row, column = (int(part) for part in bad[0])
    return [
        Finding(
            path,
            "8.4",
            f"row {row} sample {column} holds {float(stored[row, column])} "
            f"but the mean of the finite values is "
            f"{float(reference[row, column])}; {len(bad)} values differ",
        )
    ]
