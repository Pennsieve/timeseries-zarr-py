"""Runs of a continuous channel and their placement on the sample grid.

A run is a span of recorded samples one sample period apart. A source states
each run by its start time in microseconds since the Unix epoch. place_runs puts
the runs on the grid of one channel and builds the table that the bundle stores
as the runs array. Index arithmetic is exact in Fraction.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import numpy.typing as npt

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


def check_run_bound(
    first: int, time_us: int, offset_us: int, rate_hz: float
) -> None:
    """Check that a run starts within one sample period of its grid sample.

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
    period = bin_period(rate_hz, 1)
    grid_us = offset_us + first * period
    if abs(time_us - grid_us) >= period:
        raise ValueError(
            f"a run at grid sample {first} starts at {time_us} us, which is "
            f"{float(time_us - grid_us)} us from the grid time {float(grid_us)} "
            f"us. The difference must be under one sample period of "
            f"{float(period)} us"
        )


def place_runs(
    runs: Sequence[Run], rate_hz: float, reference_us: int
) -> Placement:
    """Place the runs of a channel on one sample grid.

    The first run starts at grid sample 0. Each later run starts at the grid
    sample nearest its start time, or at the sample after the previous run
    when that is later. A tie between two samples goes to the later one.

    Args:
        runs: The runs in time order.
        rate_hz: The sample rate in hertz.
        reference_us: The reference time in microseconds since the Unix epoch.

    Returns:
        The grid offset, the grid length, and the table of runs.

    Raises:
        TypeError: If a start time or a sample count is not an integer.
        ValueError: If runs is empty, rate_hz is not positive and finite, a
            run has no samples, a run starts before an earlier run, or a run
            starts before the samples of the previous run end. The last case
            means samples arrive closer together than the sample period.
    """
    period = bin_period(rate_hz, 1)
    if len(runs) == 0:
        raise ValueError("a channel needs at least one run, got none")
    offset_us = (
        integer_argument(runs[0].start_us, "run 0 start_us") - reference_us
    )
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
        if index > 0:
            if time_us < previous_time_us:
                raise ValueError(
                    f"run {index} starts at {start} us, before run "
                    f"{index - 1} at {previous_start} us. "
                    f"Runs must be in time order"
                )
            ends_us = previous_time_us + int(table[index - 1, 1]) * period
            if time_us < ends_us:
                raise ValueError(
                    f"run {index} starts at {start} us, before the samples "
                    f"of run {index - 1} end at {float(ends_us + reference_us)} "
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
