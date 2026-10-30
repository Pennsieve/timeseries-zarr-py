"""Runs of a continuous channel and their placement on the sample grid.

A run is a span of recorded samples one sample period apart. A source states
each run by its start time in microseconds since the Unix epoch. place_runs puts
the runs on the grid of one channel and builds the table that the bundle stores
as the runs array. The reader checks that table with check_runs_table and
check_outside_runs. place_runs chooses the grid index of each run in exact
Fraction arithmetic. Every bound on the table is evaluated in binary64, left
to right as docs/spec.md writes it. The writer and the reader share that
evaluation.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
import numpy.typing as npt

from timeseries_zarr.attrs import check_time_us
from timeseries_zarr.constants import MICROSECONDS_PER_SECOND, TIME_LIMIT_US
from timeseries_zarr.windows import bin_period, integer_argument


@dataclass(frozen=True, slots=True)
class Run:
    """A span of consecutive samples recorded at the sample rate.

    Attributes:
        start_us: Time of the first sample in microseconds since the Unix
            epoch.
        num_samples: Number of samples in the run, at least 1.
    """

    start_us: int
    num_samples: int


@dataclass(frozen=True, slots=True)
class Placement:
    """The runs of one channel placed on its sample grid.

    Attributes:
        offset_us: Time of grid sample 0 in microseconds from the reference
            time. It is the time of the first run.
        num_samples: Length of the grid. It counts the samples of every run
            and the NaN samples between runs.
        table: Int64 array of shape (runs, 3). Column 0 is the grid index of
            the first sample of a run, column 1 its number of samples, and
            column 2 the time of its first sample in microseconds from the
            reference time.
    """

    offset_us: int
    num_samples: int
    table: npt.NDArray[np.int64]


def _sample_period_us(rate_hz: float) -> float:
    """Return the sample period in microseconds, evaluated in binary64.

    Raises:
        ValueError: If rate_hz is not positive and finite.
    """
    if not (math.isfinite(rate_hz) and rate_hz > 0):
        raise ValueError(
            f"rate_hz must be positive and finite, got {rate_hz!r}"
        )
    return MICROSECONDS_PER_SECOND / rate_hz


def check_run_bound(
    first: int, time_us: int, offset_us: int, rate_hz: float
) -> None:
    """Check that a run starts within one sample period of its grid sample.

    The comparison is evaluated in binary64, left to right.

    Args:
        first: Grid index of the first sample of the run.
        time_us: Time of that sample in microseconds from the reference time.
        offset_us: Time of grid sample 0 in microseconds from the reference
            time.
        rate_hz: The sample rate in hertz.

    Raises:
        ValueError: If the time and the grid time of the sample differ by one
            sample period or more.
    """
    period_us = _sample_period_us(rate_hz)
    grid_us = offset_us + first * MICROSECONDS_PER_SECOND / rate_hz
    if abs(time_us - grid_us) >= period_us:
        raise ValueError(
            f"a run at grid sample {first} starts at {time_us} us, which is "
            f"{time_us - grid_us} us from the grid time {grid_us} us. The "
            f"difference must be under one sample period of {period_us} us"
        )


def first_run_end_out_of_range(
    table: npt.NDArray[np.int64], rate_hz: float
) -> tuple[int, float] | None:
    """Find the first run whose last sample lies outside the time range.

    The time of the last sample of run r is time_r + (n_r - 1) * 1e6 /
    rate_hz, evaluated in binary64 left to right. It must lie in
    (-2^53, 2^53) microseconds (docs/spec.md section 2.3).

    Args:
        table: The runs array, of shape (runs, 3).
        rate_hz: The sample rate in hertz.

    Returns:
        The row index and the time in microseconds of the first run that
        fails, or None when every run passes.
    """
    last_us = (
        table[:, 2].astype(np.float64)
        + (table[:, 1] - 1).astype(np.float64)
        * MICROSECONDS_PER_SECOND
        / rate_hz
    )
    inside = (last_us > -TIME_LIMIT_US) & (last_us < TIME_LIMIT_US)
    outside = np.flatnonzero(~inside)
    if outside.size == 0:
        return None
    row = int(outside[0])
    return row, float(last_us[row])


def place_runs(
    runs: Sequence[Run], rate_hz: float, reference_us: int
) -> Placement:
    """Place the runs of a channel on one sample grid.

    The first run starts at grid sample 0. Each later run starts at the grid
    sample nearest its start time, or at the sample after the previous run
    when that is later. A tie between two samples goes to the later one. The
    placed table satisfies the bounds of docs/spec.md section 4.2, evaluated
    in binary64.

    Args:
        runs: The runs in time order.
        rate_hz: The sample rate in hertz.
        reference_us: The reference time in microseconds since the Unix epoch.

    Returns:
        The grid offset, the grid length, and the table of runs.

    Raises:
        TypeError: If a start time or a sample count is not an integer.
        ValueError: If runs is empty, rate_hz is not positive and finite, a
            run has no samples, a run starts before an earlier run, a run
            starts before the samples of the previous run end, or a run
            time, the offset, the time of the last sample of a run, or the
            grid time of the last sample lies outside (-2^53, 2^53)
            microseconds. The third case means
            samples arrive closer together than the sample period.
    """
    period = bin_period(rate_hz, 1)
    if len(runs) == 0:
        raise ValueError("a channel needs at least one run, got none")
    offset_us = (
        integer_argument(runs[0].start_us, "run 0 start_us") - reference_us
    )
    check_time_us(offset_us, "offset_us")
    table = np.zeros((len(runs), 3), dtype=np.int64)
    previous_end = 0
    previous_time_us = 0
    previous_start = 0
    for index, run in enumerate(runs):
        start = integer_argument(run.start_us, f"run {index} start_us")
        count = integer_argument(run.num_samples, f"run {index} num_samples")
        if count < 1:
            raise ValueError(
                f"run {index} has {count} samples, which must be at least 1"
            )
        time_us = start - reference_us
        check_time_us(time_us, f"the time of run {index}")
        if index > 0:
            if time_us < previous_time_us:
                raise ValueError(
                    f"run {index} starts at {start} us, before run "
                    f"{index - 1} at {previous_start} us. "
                    f"Runs must be in time order"
                )
            ends_us = (
                previous_time_us
                + int(table[index - 1, 1]) * MICROSECONDS_PER_SECOND / rate_hz
            )
            if time_us < ends_us:
                raise ValueError(
                    f"run {index} starts at {start} us, before the samples "
                    f"of run {index - 1} end at {ends_us + reference_us} "
                    f"us. Samples must be at least one sample period of "
                    f"{float(period)} us apart"
                )
        nearest = math.floor((time_us - offset_us) / period + Fraction(1, 2))
        first = max(previous_end, nearest)
        check_run_bound(first, time_us, offset_us, rate_hz)
        table[index] = (first, count, time_us)
        previous_end = first + count
        previous_time_us = time_us
        previous_start = start
    check_time_us(
        offset_us + (previous_end - 1) * MICROSECONDS_PER_SECOND / rate_hz,
        "the grid time of the last sample",
    )
    failure = first_run_end_out_of_range(table, rate_hz)
    if failure is not None:
        raise ValueError(
            f"the last sample of run {failure[0]} is at {failure[1]} us, "
            f"which must lie in (-2^53, 2^53) microseconds"
        )
    return Placement(offset_us, previous_end, table)


def check_source_runs(
    runs: Sequence[Run], start_us: int, num_samples: int
) -> None:
    """Check that the runs of a source agree with its start and length.

    Args:
        runs: The runs the source states.
        start_us: The start time that the source reports for sample 0.
        num_samples: The number of samples that the source reports.

    Raises:
        ValueError: If runs is empty, the first run does not start at start_us,
            or the run lengths do not total num_samples.
    """
    if len(runs) == 0:
        raise ValueError("a channel needs at least one run, got none")
    if runs[0].start_us != start_us:
        raise ValueError(
            f"the first run starts at {runs[0].start_us} us, but start_us "
            f"is {start_us}. They must be equal"
        )
    total = sum(run.num_samples for run in runs)
    if total != num_samples:
        raise ValueError(
            f"the runs hold {total} samples, but num_samples is "
            f"{num_samples}. They must be equal"
        )


def _row_text(row: npt.NDArray[np.int64]) -> str:
    """Return a row of the runs table as [first, count, time_us]."""
    return f"[{int(row[0])}, {int(row[1])}, {int(row[2])}]"


def check_runs_table(
    where: str,
    table: npt.NDArray[np.int64],
    offset_us: int,
    rate_hz: float,
    num_samples: int,
) -> None:
    """Check the rows of the runs table of a channel against the spec.

    Args:
        where: The location, which begins each error message.
        table: The runs array of the channel, of shape (runs, 3).
        offset_us: The offset_us of the channel.
        rate_hz: The sample rate in hertz.
        num_samples: The length of raw.

    Raises:
        ValueError: If row 0 is not [0, n_0, offset_us], a run has no
            samples, the first samples or the times do not ascend, a run
            starts before the previous run ends, the last run does not end at
            num_samples, a run starts one sample period or more from its grid
            time, or a run starts in time before the samples of the previous
            run end. The message names the first row at fault. The last two
            bounds are evaluated in binary64, left to right as docs/spec.md
            writes them.
    """
    period_us = _sample_period_us(rate_hz)
    firsts, counts, times = table[:, 0], table[:, 1], table[:, 2]
    head = table[0]
    if int(head[0]) != 0 or int(head[2]) != offset_us:
        raise ValueError(
            f"{where}: runs row 0 is {_row_text(head)}; it must be "
            f"[0, n_0, {offset_us}], with the offset_us of the channel in "
            f"column 2"
        )
    empty = np.flatnonzero(counts < 1)
    if empty.size:
        row = int(empty[0])
        raise ValueError(
            f"{where}: runs row {row} holds {int(counts[row])} samples; "
            f"column 1 must be at least 1"
        )
    for column, noun, values in (
        (0, "first sample", firsts),
        (2, "time", times),
    ):
        stalled = np.flatnonzero(np.diff(values) <= 0)
        if stalled.size:
            row = int(stalled[0]) + 1
            raise ValueError(
                f"{where}: runs row {row} has {noun} {int(values[row])}, "
                f"which is not above {int(values[row - 1])} of row "
                f"{row - 1}; column {column} must ascend"
            )
    ends = firsts + counts
    overlapping = np.flatnonzero(firsts[1:] < ends[:-1])
    if overlapping.size:
        row = int(overlapping[0]) + 1
        raise ValueError(
            f"{where}: runs row {row} starts at sample {int(firsts[row])}, "
            f"inside run {row - 1}, which ends at sample "
            f"{int(ends[row - 1])}; the first sample must be at least "
            f"{int(ends[row - 1])}"
        )
    if int(ends[-1]) != num_samples:
        raise ValueError(
            f"{where}: the last run, row {len(table) - 1}, ends at sample "
            f"{int(ends[-1])}, but raw holds {num_samples} samples; the last "
            f"run must end at the end of raw"
        )
    grid_us = (
        offset_us
        + firsts.astype(np.float64) * MICROSECONDS_PER_SECOND / rate_hz
    )
    deviation = np.abs(times.astype(np.float64) - grid_us)
    late = np.flatnonzero(deviation >= period_us)
    if late.size:
        row = int(late[0])
        raise ValueError(
            f"{where}: runs row {row} has time {int(times[row])} us, "
            f"which is {float(times[row]) - float(grid_us[row])} us from the "
            f"grid time {float(grid_us[row])} us of sample "
            f"{int(firsts[row])}; the difference must be under one sample "
            f"period of {period_us} us"
        )
    ends_us = (
        times[:-1].astype(np.float64)
        + counts[:-1].astype(np.float64) * MICROSECONDS_PER_SECOND / rate_hz
    )
    early = np.flatnonzero(times[1:] < ends_us)
    if early.size:
        index = int(early[0])
        raise ValueError(
            f"{where}: runs row {index + 1} has time "
            f"{int(times[index + 1])} us, before the samples of run "
            f"{index} end at {float(ends_us[index])} us; the time must be at "
            f"least {math.ceil(ends_us[index])} us"
        )


def check_outside_runs(
    where: str,
    values: npt.NDArray[np.floating[Any]],
    first: int,
    table: npt.NDArray[np.int64],
) -> None:
    """Check that the raw samples that lie outside every run are NaN.

    Args:
        where: The location, which begins the error message.
        values: Raw samples. Element i is sample first + i.
        first: The index of values[0].
        table: The runs table of the channel.

    Raises:
        ValueError: If a sample outside every run is not NaN. The message
            names the first such sample.
    """
    indices = np.arange(first, first + values.shape[0], dtype=np.int64)
    rows = np.maximum(
        np.searchsorted(table[:, 0], indices, side="right") - 1, 0
    )
    outside = indices >= table[rows, 0] + table[rows, 1]
    bad = np.flatnonzero(outside & ~np.isnan(values))
    if bad.size:
        index = int(indices[bad[0]])
        raise ValueError(
            f"{where}: raw sample {index} lies outside every run and holds "
            f"{float(values[bad[0]])}; it must be NaN"
        )
