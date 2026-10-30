"""Write the conformance fixtures of docs/spec.md and their manifest.

Usage: python scripts/build_fixtures.py [OUTPUT_DIRECTORY]

The default directory is conformance/ at the root of the repository. Each
fixture is one small bundle in its own directory. manifest.json lists what each
fixture exercises, the values a reader can assert, and the findings a validator
reports. The values come from the arrays in this script, not from reading a
bundle back. The output does not depend on the date, the machine, or the
thread count.
"""

import argparse
import json
import math
import shutil
import sys
import warnings
import zlib
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import zarr
from zarr.core.common import JSON

from timeseries_zarr.attrs import (
    continuous_channel_attrs,
    event_channel_attrs,
    level_group_attrs,
    root_attrs,
    waveform_array_attrs,
)
from timeseries_zarr.bundle import write_bundle
from timeseries_zarr.recording import write_recording_json
from timeseries_zarr.runs import Run
from timeseries_zarr.types import WriteOpts
from timeseries_zarr.zarr_io import (
    ZarrGroup,
    consolidate,
    create_array,
    create_group_with_attrs,
    open_group,
    write_region,
)

SEED = 20261029
"""Base seed. Each fixture adds the checksum of its name."""
SPEC_VERSION = "1.0"
EXTENSION_VERSION = "1.1"
INNER_ROWS = 256
SHARD_ROWS = 1024
ZSTD_LEVEL = 3
FACTOR = 4
MAX_LEVELS = 15
TIME_LIMIT_US = 2**53
MICROSECONDS_PER_SECOND = 1e6
WRITE_OPTS = WriteOpts(
    zstd_level=ZSTD_LEVEL,
    min_bins=1,
    inner_len=INNER_ROWS,
    target_shard_bytes=4 * SHARD_ROWS,
    max_workers=1,
)
SESSION_START = datetime(
    2024, 3, 14, 9, 15, tzinfo=timezone(timedelta(hours=-5))
)
REFERENCE_TIME = SESSION_START.astimezone(UTC)
RATE_30KHZ = 30000.0
RATE_TDT_HZ = 24414.0625
RATE_1KHZ = 1000.0

type Entry = dict[str, Any]
type Build = Callable[[Path], Entry]


def _generator(name: str) -> np.random.Generator:
    """Return the seeded generator of the fixture name."""
    return np.random.default_rng([SEED, zlib.crc32(name.encode("ascii"))])


def _number(value: float) -> float | None:
    """Return value as a JSON number, or None for NaN.

    Raises:
        ValueError: If value is an infinity, which no fixture holds.
    """
    if math.isnan(value):
        return None
    if math.isinf(value):
        raise ValueError("a manifest value must not be infinite")
    return float(value)


def _noise(
    name: str, length: int, centre: float, spread: float
) -> npt.NDArray[np.float32]:
    """Return length float32 samples of seeded noise."""
    values = centre + spread * _generator(name).standard_normal(length)
    return values.astype(np.float32)


def _level_widths(length: int) -> list[int]:
    """Return the samples per bin of each level the writer keeps.

    Level k stays while it holds at least one bin, up to MAX_LEVELS levels.
    """
    widths = (FACTOR**level for level in range(1, MAX_LEVELS + 1))
    return [width for width in widths if width <= length]


class SignalSource:
    """A continuous source over an array of samples."""

    unit = "uV"

    def __init__(
        self, samples: npt.NDArray[np.float32], name: str, rate_hz: float
    ) -> None:
        """Hold the samples, the name, which is also the id, and the rate."""
        self._samples = samples
        self.id = name
        self.name = name
        self._rate_hz = rate_hz

    def rate_hz(self) -> float:
        """Return the sample rate."""
        return self._rate_hz

    def start_us(self) -> int:
        """Return the start time, the reference time."""
        return 0

    def num_samples(self) -> int:
        """Return the number of samples."""
        return int(self._samples.shape[0])

    def read_samples(self, start: int, stop: int) -> npt.NDArray[np.float32]:
        """Return the samples from start up to stop."""
        return self._samples[start:stop]


class RunsSource(SignalSource):
    """A continuous source that states the runs of its samples."""

    def __init__(
        self,
        runs: Sequence[tuple[int, npt.NDArray[np.float32]]],
        name: str,
        rate_hz: float,
    ) -> None:
        """Hold each run as its start time and its samples."""
        self._starts = [start for start, _ in runs]
        super().__init__(
            np.concatenate([samples for _, samples in runs]), name, rate_hz
        )
        self._lengths = [int(samples.shape[0]) for _, samples in runs]

    def start_us(self) -> int:
        """Return the start time of the first run."""
        return self._starts[0]

    def runs(self) -> list[Run]:
        """Return the runs in time order."""
        return [
            Run(start, length)
            for start, length in zip(self._starts, self._lengths, strict=True)
        ]


def _bin_statistics(
    grid: npt.NDArray[np.float32], width: int, index: int, with_var: bool
) -> Entry:
    """Return the statistics of bin index of the level of width samples.

    A bin without a finite sample has count 0 and null statistics.
    """
    block = grid[index * width : (index + 1) * width].astype(np.float64)
    finite = block[np.isfinite(block)]
    entry: Entry = {"index": index, "count": int(finite.shape[0])}
    if not finite.shape[0]:
        entry.update(min=None, max=None, mean=None)
        if with_var:
            entry["var"] = None
        return entry
    entry.update(
        min=float(finite.min()),
        max=float(finite.max()),
        mean=float(finite.mean()),
    )
    if with_var:
        entry["var"] = float(finite.var())
    return entry


def _continuous_entry(
    index: int,
    grid: npt.NDArray[np.float32],
    *,
    name: str,
    rate_hz: float,
    widths: Sequence[int],
    probes: Sequence[int] = (),
    runs: Sequence[Sequence[int]] | None = None,
    with_var: bool = False,
) -> Entry:
    """Return the manifest entry of a continuous channel.

    Args:
        index: The channel index.
        grid: The expected content of raw, NaN outside every run.
        name: The channel name and id.
        rate_hz: The sample rate.
        widths: The samples per bin of each level the channel holds.
        probes: Sample indices to report beside the first, the middle, and
            the last. The level bins that hold them are reported too.
        runs: The expected runs table, or None for a channel without it.
        with_var: Whether each level holds var.
    """
    length = int(grid.shape[0])
    indices = sorted(
        {
            index
            for index in (0, 1, length // 2, length - 1)
            if 0 <= index < length
        }
        | set(probes)
    )
    members = ["minmax", "mean", "count"] + (["var"] if with_var else [])
    levels: list[Entry] = []
    for number, width in enumerate(widths, start=1):
        bins = -(-length // width)
        chosen = sorted({0, bins // 2, bins - 1} | {i // width for i in probes})
        levels.append(
            {
                "level": number,
                "samples_per_bin": width,
                "period_us": width * MICROSECONDS_PER_SECOND / rate_hz,
                "bins": bins,
                "members": members,
                "bin_values": [
                    _bin_statistics(grid, width, index, with_var)
                    for index in chosen
                ],
            }
        )
    return {
        "index": index,
        "kind": "continuous",
        "name": name,
        "id": name,
        "unit": "uV",
        "rate_hz": rate_hz,
        "offset_us": 0,
        "num_samples": length,
        "runs": None if runs is None else [list(row) for row in runs],
        "samples": [
            {"index": index, "value": _number(grid[index])} for index in indices
        ],
        "levels": levels,
    }


def _fixture(
    name: str,
    exercises: str,
    channels: Sequence[Entry] | None,
    *,
    recording: Entry | None = None,
    version: str = SPEC_VERSION,
    findings: Sequence[Entry] = (),
    reader_rejects: bool = False,
) -> Entry:
    """Return the manifest entry of a fixture.

    A fixture whose channels is None gives no channel values.
    """
    entry: Entry = {
        "path": name,
        "exercises": exercises,
        "version": version,
        "recording_json": recording,
        "reader_must_reject": reader_rejects,
        "findings": list(findings),
    }
    if channels is not None:
        entry["channel_count"] = len(channels)
        entry["channels"] = list(channels)
    return entry


def _write_continuous(
    out: Path,
    name: str,
    exercises: str,
    grid: npt.NDArray[np.float32],
    *,
    rate_hz: float,
    probes: Sequence[int] = (),
    with_var: bool = False,
) -> Entry:
    """Write a bundle of one continuous channel through the writer."""
    source = SignalSource(grid, name, rate_hz)
    opts = replace(WRITE_OPTS, write_var=with_var)
    write_bundle([source], [], output_path=out / name, opts=opts)
    channel = _continuous_entry(
        0,
        grid,
        name=name,
        rate_hz=rate_hz,
        widths=_level_widths(int(grid.shape[0])),
        probes=probes,
        with_var=with_var,
    )
    return _fixture(name, exercises, [channel])


def _build_30khz(out: Path) -> Entry:
    """Write the 30 kHz channel."""
    name = "continuous_30khz"
    grid = _noise(name, 6000, 0.0, 100.0)
    return _write_continuous(
        out,
        name,
        "A channel at 30000 Hz, where the period of a level is not a whole "
        "number of microseconds. The last bin of levels 3 through 6 holds "
        "fewer samples than the width of its level.",
        grid,
        rate_hz=RATE_30KHZ,
        probes=(5999,),
    )


def _build_tdt(out: Path) -> Entry:
    """Write the 24414.0625 Hz channel."""
    name = "continuous_24414hz"
    grid = _noise(name, 5000, 0.0, 100.0)
    return _write_continuous(
        out,
        name,
        "A channel at 24414.0625 Hz, a rate that is exact in binary and has "
        "a sample period of 40.96 microseconds. The last bin of levels 2 "
        "through 6 holds fewer samples than the width of its level.",
        grid,
        rate_hz=RATE_TDT_HZ,
        probes=(4999,),
    )


RUN_TABLE = [[0, 1500, 0], [2200, 1000, 2_200_400], [3200, 800, 3_200_450]]
"""The runs of the sparse channel: first sample, samples, time of the first."""


def _build_runs(out: Path) -> Entry:
    """Write the channel with three runs and a pause."""
    name = "continuous_runs"
    rng = _generator(name)
    parts = [
        (100.0 * (index + 1) + 8.0 * rng.standard_normal(length)).astype(
            np.float32
        )
        for index, (_, length, _) in enumerate(RUN_TABLE)
    ]
    source = RunsSource(
        [(row[2], part) for row, part in zip(RUN_TABLE, parts, strict=True)],
        name,
        RATE_1KHZ,
    )
    write_bundle([source], [], output_path=out / name, opts=WRITE_OPTS)
    length = RUN_TABLE[-1][0] + RUN_TABLE[-1][1]
    grid = np.full(length, np.nan, dtype=np.float32)
    for (first, count, _), part in zip(RUN_TABLE, parts, strict=True):
        grid[first : first + count] = part
    channel = _continuous_entry(
        0,
        grid,
        name=name,
        rate_hz=RATE_1KHZ,
        widths=_level_widths(length),
        probes=(1499, 1500, 2199, 2200, 3199, 3200),
        runs=RUN_TABLE,
    )
    return _fixture(
        name,
        "A channel of three runs at 1000 Hz. A pause of 700 samples between "
        "the first two leaves NaN samples, so bins of levels 1 and 2 hold "
        "count 0 and one bin holds a partial count. The second run starts "
        "0.4 sample periods off the grid. The third run starts 0.05 sample "
        "periods after the end of the second, which leaves no NaN sample.",
        [channel],
    )


DROPOUT_INDEX = 1234


def _build_dropout(out: Path) -> Entry:
    """Write the channel with one NaN sample."""
    name = "continuous_dropout"
    grid = _noise(name, 4096, 0.0, 100.0)
    grid[DROPOUT_INDEX] = np.nan
    return _write_continuous(
        out,
        name,
        "A channel of 4096 samples at 1000 Hz with one NaN sample at index "
        "1234. The sample leaves count one below the width in the bin that "
        "holds it at each of the 6 levels, and enters no minimum, maximum, "
        "or mean.",
        grid,
        rate_hz=RATE_1KHZ,
        probes=(DROPOUT_INDEX - 1, DROPOUT_INDEX, DROPOUT_INDEX + 1),
    )


def _build_dc_offset(out: Path) -> Entry:
    """Write the channel with a mean of 1e4."""
    name = "continuous_dc_offset"
    grid = _noise(name, 4096, 1e4, 5.0)
    return _write_continuous(
        out,
        name,
        "A channel of 4096 samples at 1000 Hz with mean 1e4 and standard "
        "deviation 5, whose levels hold var. A variance folded from float32 "
        "means loses its precision at this offset.",
        grid,
        rate_hz=RATE_1KHZ,
        with_var=True,
    )


def _build_empty_continuous(out: Path) -> Entry:
    """Write the channel with no samples."""
    name = "continuous_empty"
    return _write_continuous(
        out,
        name,
        "A continuous channel with N = 0: raw has shape (0,), and the "
        "channel holds no runs and no levels.",
        np.zeros(0, dtype=np.float32),
        rate_hz=RATE_30KHZ,
    )


def _sort_consolidated(path: Path) -> None:
    """Sort the consolidated metadata of the bundle at path by node path.

    The order of the entries carries no meaning, and the order that the
    listing of a directory gives differs between filesystems.
    """
    file = path / "zarr.json"
    document = json.loads(file.read_text(encoding="utf-8"))
    consolidated = document.get("consolidated_metadata")
    if consolidated is not None:
        entries = consolidated["metadata"]
        consolidated["metadata"] = {
            key: entries[key] for key in sorted(entries)
        }
    file.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _write_array(
    group: ZarrGroup,
    name: str,
    data: npt.NDArray[Any],
    dimensions: Sequence[str | None],
    attrs: dict[str, object] | None = None,
) -> None:
    """Create the array name under group and write data into it."""
    trailing = data.shape[1:]
    array = create_array(
        group,
        name,
        data.shape,
        data.dtype,
        (INNER_ROWS, *trailing),
        (SHARD_ROWS, *trailing),
        {} if attrs is None else attrs,
        ZSTD_LEVEL,
        dimension_names=dimensions,
    )
    for start in range(0, data.shape[0], SHARD_ROWS):
        write_region(array, start, data[start : start + SHARD_ROWS])


def _write_raw_channel(
    root: ZarrGroup, index: int, grid: npt.NDArray[np.float32], name: str
) -> Entry:
    """Write a continuous channel that holds raw and no level, and describe it."""
    channel = create_group_with_attrs(
        root,
        str(index),
        continuous_channel_attrs(name, name, RATE_1KHZ, 0, "uV", None),
    )
    _write_array(channel, "raw", grid, ("time",))
    return _continuous_entry(
        index, grid, name=name, rate_hz=RATE_1KHZ, widths=()
    )


def _encode_text(texts: Sequence[str], width: int) -> npt.NDArray[np.uint8]:
    """Return the rows of UTF-8 bytes of texts, padded with zeros to width."""
    rows = np.zeros((len(texts), width), dtype=np.uint8)
    for index, text in enumerate(texts):
        encoded = text.encode("utf-8")
        rows[index, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return rows


def _encode_channels(
    sets: Sequence[Sequence[int]], width: int
) -> npt.NDArray[np.uint8]:
    """Return the bitmask rows of sets, least significant bit first."""
    rows = np.zeros((len(sets), width), dtype=np.uint8)
    for row, indices in enumerate(sets):
        for index in indices:
            rows[row, index // 8] |= 1 << (index % 8)
    return rows


def _column_numbers(values: npt.NDArray[Any]) -> list[Any]:
    """Return values as a nested list of JSON numbers, null for NaN."""
    if values.ndim > 1:
        return [_column_numbers(row) for row in values]
    if np.issubdtype(values.dtype, np.floating):
        return [_number(float(value)) for value in values]
    return [int(value) for value in values]


def _waveform_means(
    waveforms: npt.NDArray[np.float32],
    labels: npt.NDArray[np.uint16],
    label_count: int,
) -> npt.NDArray[np.float32]:
    """Return the mean over the finite values of the rows of each label.

    A sample with no finite value, and a label with no event, give NaN.
    """
    means = np.full((label_count, waveforms.shape[1]), np.nan)
    for label in range(label_count):
        rows = waveforms[labels == label].astype(np.float64)
        for sample in range(rows.shape[1]):
            finite = rows[np.isfinite(rows[:, sample]), sample]
            if finite.shape[0]:
                means[label, sample] = finite.mean()
    return means.astype(np.float32)


class EventColumns:
    """The arrays of one event channel and the metadata that describes them."""

    def __init__(
        self,
        events: npt.NDArray[np.int64],
        *,
        label_names: list[str],
        max_duration_us: int | None = None,
        durations: npt.NDArray[np.int64] | None = None,
        labels: npt.NDArray[np.uint16] | None = None,
        values: npt.NDArray[np.float32] | None = None,
        texts: Sequence[str] | None = None,
        channel_sets: Sequence[Sequence[int]] | None = None,
        waveforms: npt.NDArray[np.float32] | None = None,
        waveform_period_us: float = 1e6 / RATE_30KHZ,
        event_sample: int = 0,
        count_periods: Sequence[int] = (),
    ) -> None:
        """Hold the columns. None leaves a column out of the channel."""
        self.events = events
        self.label_names = label_names
        self.max_duration_us = max_duration_us
        self.durations = durations
        self.labels = labels
        self.values = values
        self.texts = texts
        self.channel_sets = channel_sets
        self.waveforms = waveforms
        self.waveform_period_us = waveform_period_us
        self.event_sample = event_sample
        self.count_periods = count_periods


def _count_level(
    events: npt.NDArray[np.int64],
    labels: npt.NDArray[np.uint16] | None,
    label_count: int,
    period_us: int,
) -> tuple[int, npt.NDArray[np.uint32]]:
    """Return the offset and the counts of one count level, by section 5.7."""
    offset_us = int(events[0]) // period_us * period_us
    bins = (int(events[-1]) - offset_us) // period_us + 1
    counts = np.zeros((bins, label_count), dtype=np.uint32)
    column = np.zeros(events.shape, dtype=np.int64)
    if labels is not None:
        column = labels.astype(np.int64)
    np.add.at(counts, ((events - offset_us) // period_us, column), 1)
    return offset_us, counts


def _write_event_channel(
    root: ZarrGroup,
    index: int,
    name: str,
    channel_count: int,
    columns: EventColumns,
) -> Entry:
    """Write an event channel by hand and return its manifest entry.

    Args:
        root: The root group of the bundle.
        index: The channel index.
        name: The channel name and id.
        channel_count: The channels in the bundle, which sets the width of the
            channels column.
        columns: The arrays to write.
    """
    count = int(columns.events.shape[0])
    label_names = columns.label_names
    label_count = len(label_names)
    attrs = event_channel_attrs(
        name,
        name,
        "uV",
        label_names,
        columns.max_duration_us,
        {"detector": "threshold", "version": "1"},
    )
    channel = create_group_with_attrs(root, str(index), attrs)
    entry: Entry = {
        "index": index,
        "kind": "event",
        "name": name,
        "id": name,
        "unit": "uV",
        "label_names": label_names,
        "num_labels": label_count,
        "max_duration_us": columns.max_duration_us,
        "num_events": count,
        "events": [int(time) for time in columns.events],
    }
    column_names: list[str] = []
    _write_array(channel, "events", columns.events, ("event",))
    if columns.durations is not None:
        _write_array(channel, "durations", columns.durations, ("event",))
        entry["durations"] = _column_numbers(columns.durations)
        column_names.append("durations")
    if columns.labels is not None:
        _write_array(channel, "labels", columns.labels, ("event",))
        entry["labels"] = _column_numbers(columns.labels)
        column_names.append("labels")
    if columns.values is not None:
        _write_array(channel, "values", columns.values, ("event",))
        entry["values"] = _column_numbers(columns.values)
        column_names.append("values")
    if columns.texts is not None:
        width = max(
            1, max((len(t.encode("utf-8")) for t in columns.texts), default=0)
        )
        _write_array(
            channel, "text", _encode_text(columns.texts, width), ("event", None)
        )
        entry["text_width"] = width
        entry["text"] = list(columns.texts)
        column_names.append("text")
    if columns.channel_sets is not None:
        width = (channel_count + 7) // 8
        rows = _encode_channels(columns.channel_sets, width)
        _write_array(channel, "channels", rows, ("event", None))
        entry["channel_bytes"] = width
        entry["channels"] = [list(indices) for indices in columns.channel_sets]
        column_names.append("channels")
    if columns.waveforms is not None:
        assert columns.labels is not None
        samples = columns.waveforms.shape[1]
        attrs = waveform_array_attrs(
            columns.waveform_period_us, columns.event_sample, samples
        )
        _write_array(
            channel, "waveforms", columns.waveforms, ("event", None), attrs
        )
        means = _waveform_means(columns.waveforms, columns.labels, label_count)
        _write_array(channel, "waveform_mean", means, ("label", None), attrs)
        entry["waveform"] = {
            "samples": samples,
            "period_us": columns.waveform_period_us,
            "event_sample": columns.event_sample,
            "waveforms": _column_numbers(columns.waveforms),
            "waveform_mean": _column_numbers(means),
        }
        column_names.append("waveforms")
    entry["columns"] = column_names
    entry["count_levels"] = []
    for number, period in enumerate(columns.count_periods, start=1):
        offset, counts = _count_level(
            columns.events, columns.labels, label_count, period
        )
        level = create_group_with_attrs(
            channel, str(number), level_group_attrs(period, offset)
        )
        _write_array(level, "count", counts, ("time", None))
        entry["count_levels"].append(
            {
                "level": number,
                "period_us": period,
                "offset_us": offset,
                "bins": int(counts.shape[0]),
                "count": _column_numbers(counts),
            }
        )
    return entry


def _build_empty_events(out: Path) -> Entry:
    """Write the event channel with no event."""
    name = "events_empty"
    root = open_group(out / name, root_attrs())
    columns = EventColumns(
        np.zeros(0, dtype=np.int64),
        label_names=["first", "second"],
        max_duration_us=0,
        durations=np.zeros(0, dtype=np.int64),
        labels=np.zeros(0, dtype=np.uint16),
        values=np.zeros(0, dtype=np.float32),
        texts=[],
        channel_sets=[],
        waveforms=np.zeros((0, 8), dtype=np.float32),
    )
    entry = _write_event_channel(root, 0, name, 1, columns)
    consolidate(root)
    return _fixture(
        name,
        "An event channel with n = 0 that holds every optional column except "
        "a count level. Each column has 0 rows, text has width 1, channels "
        "has 1 byte, and waveform_mean holds L = 2 rows of NaN.",
        [entry],
    )


EVENT_COUNT = 24
EVENT_TEXTS = [
    "",
    "caf\N{LATIN SMALL LETTER E WITH ACUTE}",
    "\N{CJK UNIFIED IDEOGRAPH-6E2C}\N{CJK UNIFIED IDEOGRAPH-8A66}",
    "burst",
    "\N{GRINNING FACE} onset",
]
EVENT_CHANNEL_SETS = [
    [0],
    [1, 8],
    [],
    list(range(10)),
    [8],
]
EVENT_PERIODS_US = (16_000, 64_000, 256_000)
EVENT_CHANNELS = 10


def _build_events(out: Path) -> Entry:
    """Write the event channel that holds every column."""
    name = "events_all_columns"
    rng = _generator(name)
    root = open_group(out / name, root_attrs())
    entries = [
        _write_raw_channel(
            root,
            index,
            _noise(f"{name}-{index}", 64, 0.0, 100.0),
            f"raw-{index}",
        )
        for index in range(EVENT_CHANNELS - 1)
    ]
    index = np.arange(EVENT_COUNT)
    events = (-40_000 + 16_000 * index + (index * index % 7) * 100).astype(
        np.int64
    )
    events[7] = events[6]
    labels = (index % 3 == 0).astype(np.uint16)
    durations = np.where(index % 5 == 0, 250_000 * (1 + index % 3), 0).astype(
        np.int64
    )
    waveforms = rng.standard_normal((EVENT_COUNT, 8)).astype(np.float32)
    waveforms[3, 2] = np.nan
    waveforms[labels == 1, 5] = np.nan
    columns = EventColumns(
        events,
        label_names=["spike", "burst", "artifact"],
        max_duration_us=2_000_000,
        durations=durations,
        labels=labels,
        values=((index - 8) * 0.25).astype(np.float32),
        texts=[EVENT_TEXTS[i % len(EVENT_TEXTS)] for i in range(EVENT_COUNT)],
        channel_sets=[
            EVENT_CHANNEL_SETS[i % len(EVENT_CHANNEL_SETS)]
            for i in range(EVENT_COUNT)
        ],
        waveforms=waveforms,
        event_sample=2,
        count_periods=EVENT_PERIODS_US,
    )
    entries.append(
        _write_event_channel(
            root, EVENT_CHANNELS - 1, "events", EVENT_CHANNELS, columns
        )
    )
    consolidate(root)
    recording: Entry = {
        "reference_time": REFERENCE_TIME.isoformat(timespec="microseconds"),
        "session": {
            "start_time": SESSION_START.isoformat(timespec="microseconds")
        },
    }
    write_recording_json(out / name, recording)
    return _fixture(
        name,
        "An event channel at index 9 of a bundle of 10 channels, so channels "
        "is 2 bytes wide. It holds durations below a loose max_duration_us, "
        "a tie in the event times, labels 0 and 1 of 3 label_names, values, "
        "text with multi-byte UTF-8, an empty text, and a text as wide as "
        "the column, channels with a row that names no channel and a row "
        "with the last valid bit set, waveforms and waveform_mean with a "
        "label that has no event and a sample with no finite value, and 3 "
        "count levels whose offset_us is negative. recording.json holds a "
        "reference_time and a session start_time in another zone.",
        entries,
        recording=recording,
    )


def _tsz(group: ZarrGroup) -> dict[str, Any]:
    """Return a copy of the tsz attributes of group."""
    return dict(cast("dict[str, Any]", group.attrs["tsz"]))


def _add_tsz_key(group: ZarrGroup, key: str, value: object) -> None:
    """Set key in the tsz attributes of group."""
    group.attrs["tsz"] = cast("dict[str, JSON]", {**_tsz(group), key: value})


def _child_group(group: ZarrGroup, name: str) -> ZarrGroup:
    """Return the child group name of group."""
    child = group[name]
    assert isinstance(child, zarr.Group)
    return child


def _build_extension(out: Path) -> Entry:
    """Write the 1.1 style bundle."""
    name = "extension_1_1"
    path = out / name
    grid = _noise(name, 1024, 0.0, 100.0)
    source = SignalSource(grid, "wide", RATE_1KHZ)
    write_bundle([source], [], output_path=path, opts=WRITE_OPTS)
    root = zarr.open_group(path, mode="r+", use_consolidated=False)
    root.attrs["tsz"] = {
        "version": EXTENSION_VERSION,
        "future_key": "a key that 1.0 does not define",
    }
    channel = _child_group(root, "0")
    _add_tsz_key(channel, "future_key", {"nested": [1, 2, 3]})
    level = _child_group(channel, "1")
    _add_tsz_key(level, "future_key", 7)
    bins = -(-int(grid.shape[0]) // FACTOR)
    _write_array(level, "median", np.zeros(bins, dtype=np.float32), ("time",))
    create_group_with_attrs(channel, "notes", {"tsz": {"kind": "note"}})
    create_group_with_attrs(root, "montages", {"montage": {"pairs": [[0, 1]]}})
    index = np.arange(8)
    events = EventColumns(
        (1_000_000 * index).astype(np.int64),
        label_names=["first", "second"],
        labels=(index % 2).astype(np.uint16),
    )
    entry = _write_event_channel(root, 1, "marks", 2, events)
    marks = _child_group(root, "1")
    _add_tsz_key(marks, "future_key", "x")
    _write_array(
        marks, "confidence", np.full(8, 0.5, dtype=np.float32), ("event",)
    )
    consolidate(root)
    (path / "NOTICE.txt").write_text("A file that 1.0 does not define.\n")
    (path / "0" / "README.txt").write_text("Another file.\n")
    recording: Entry = {
        "reference_time": REFERENCE_TIME.isoformat(timespec="microseconds"),
        "device": {"model": "a key that 1.0 does not define"},
    }
    write_recording_json(path, recording)
    channel_entry = _continuous_entry(
        0,
        grid,
        name="wide",
        rate_hz=RATE_1KHZ,
        widths=_level_widths(1024),
    )
    return _fixture(
        name,
        "A bundle that declares version 1.1 and holds what a later minor "
        "release may add: an unknown tsz key on the root, a channel, a level, "
        "and an event channel, an unknown group under a channel, an unknown "
        "member of a level, an unknown column of an event channel, an "
        "unknown node under the root, an unknown key in recording.json, and "
        "two unknown files. A 1.0 reader reads it and ignores them. A "
        "validator reports the minor version as unsupported and nothing "
        "else.",
        [channel_entry, entry],
        recording=recording,
        version=EXTENSION_VERSION,
        findings=[
            {
                "path": "",
                "section": "9",
                "message_contains": "minor version",
            }
        ],
    )


def _minimal(
    path: Path, events: npt.NDArray[np.int64]
) -> tuple[ZarrGroup, Entry]:
    """Write a bundle of one continuous and one event channel, unconsolidated."""
    root = open_group(path, root_attrs())
    grid = _noise(path.name, 64, 0.0, 100.0)
    _write_raw_channel(root, 0, grid, "raw")
    columns = EventColumns(
        events,
        label_names=["first", "second"],
        labels=(np.arange(events.shape[0]) % 2).astype(np.uint16),
    )
    return root, _write_event_channel(root, 1, "marks", 2, columns)


def _build_no_consolidated(out: Path) -> Entry:
    """Write the bundle whose root has no consolidated metadata."""
    name = "invalid/no_consolidated_metadata"
    path = out / name
    root, _ = _minimal(path, np.arange(8, dtype=np.int64) * 1000)
    consolidate(root)
    file = path / "zarr.json"
    document = json.loads(file.read_text(encoding="utf-8"))
    del document["consolidated_metadata"]
    file.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return _fixture(
        name,
        "A root zarr.json without consolidated_metadata, which is not a "
        "bundle. A reader rejects it.",
        None,
        findings=[
            {
                "path": "",
                "section": "6.2",
                "message_contains": "no consolidated_metadata",
            }
        ],
        reader_rejects=True,
    )


def _build_missing_chunk(out: Path) -> Entry:
    """Write the bundle whose events array has no chunk."""
    name = "invalid/missing_events_chunk"
    path = out / name
    root, _ = _minimal(path, np.arange(8, dtype=np.int64) * 1000)
    consolidate(root)
    shutil.rmtree(path / "1" / "events" / "c")
    return _fixture(
        name,
        "An event channel of 8 events whose events array has no chunk. "
        "Section 7.3 requires every chunk of a per-event column.",
        None,
        findings=[
            {
                "path": "1/events",
                "section": "7.3",
                "message_contains": "chunks are missing",
            }
        ],
    )


def _build_time_limit(out: Path) -> Entry:
    """Write the bundle with an event time at 2^53."""
    name = "invalid/event_time_past_limit"
    path = out / name
    events = np.array([0, 1000, TIME_LIMIT_US], dtype=np.int64)
    root, _ = _minimal(path, events)
    consolidate(root)
    return _fixture(
        name,
        "An event at 2^53 microseconds, which section 2.3 excludes. The "
        "writer refuses such a time, so the bundle is built by hand.",
        None,
        findings=[
            {
                "path": "1/events",
                "section": "2.3",
                "message_contains": "event time",
            }
        ],
    )


BUILDERS: list[Build] = [
    _build_30khz,
    _build_tdt,
    _build_runs,
    _build_dropout,
    _build_dc_offset,
    _build_empty_continuous,
    _build_empty_events,
    _build_events,
    _build_extension,
    _build_no_consolidated,
    _build_missing_chunk,
    _build_time_limit,
]

CONVENTIONS = [
    "Every path is relative to this directory. Each fixture is a bundle.",
    "A null number stands for NaN. No fixture holds an infinity.",
    "min, max, count, and the integer values compare exactly. mean and var "
    "compare within 2^-20 relative.",
    "period_us of a level is the value of the level attribute, and "
    "samples_per_bin is the width s_k of section 4.3.",
    "columns lists the per-event columns of an event channel other than "
    "events, in the order of section 5.1.",
    "recording_json is the content of recording.json, or null when the "
    "bundle has none.",
    "findings lists what full validation reports, as a path, a section, and a "
    "fragment of the message. A conforming bundle has none.",
    "reader_must_reject marks a bundle that section 8.2 requires a reader to "
    "reject. A reader may accept the other invalid bundles.",
]


def main(argv: Sequence[str] | None = None) -> int:
    """Write every fixture and the manifest.

    Args:
        argv: The command line arguments. None reads sys.argv.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "conformance",
        help="the directory to write, replaced when it holds a manifest",
    )
    out = parser.parse_args(argv).output
    if out.exists() and any(out.iterdir()):
        if not (out / "manifest.json").is_file():
            print(
                f"{out} is not empty and holds no manifest.json",
                file=sys.stderr,
            )
            return 1
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings(
        "ignore",
        message="Consolidated metadata is currently not part",
        category=UserWarning,
    )
    fixtures: dict[str, Entry] = {}
    for build in BUILDERS:
        entry = build(out)
        _sort_consolidated(out / entry["path"])
        fixtures[entry["path"]] = entry
    manifest = {
        "spec_version": SPEC_VERSION,
        "conventions": CONVENTIONS,
        "fixtures": fixtures,
    }
    text = json.dumps(manifest, indent=1)
    (out / "manifest.json").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
