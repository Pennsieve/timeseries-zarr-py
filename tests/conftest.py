"""Shared pytest fixtures, the in-memory sources, and the test bundles."""

import json
import warnings

import numpy as np
import pytest
from zarr.buffer import default_buffer_prototype
from zarr.storage import MemoryStore

from tests.test_bundle import (
    _RECORDING,
    _SESSION_US,
    _SPEC_OPTS,
    _mixed_sources,
)
from timeseries_zarr.attrs import (
    continuous_channel_attrs,
    event_channel_attrs,
    level_group_attrs,
    root_attrs,
    waveform_array_attrs,
)
from timeseries_zarr.bundle import write_bundle
from timeseries_zarr.columns import channels_rows, text_rows, text_width
from timeseries_zarr.constants import (
    CHANNELS_KEY,
    COUNT_KEY,
    DEFAULT_DECIMATION_FACTOR,
    DURATIONS_KEY,
    EVENT_DIMENSION,
    EVENTS_KEY,
    LABEL_DIMENSION,
    LABELS_KEY,
    MEAN_KEY,
    MINMAX_KEY,
    RAW_KEY,
    RUNS_KEY,
    TEXT_KEY,
    TIME_DIMENSION,
    TSZ_KEY,
    VALUES_KEY,
    VAR_KEY,
    WAVEFORM_MEAN_KEY,
    WAVEFORMS_KEY,
)
from timeseries_zarr.planning import level_period_us
from timeseries_zarr.runs import Run
from timeseries_zarr.zarr_io import (
    consolidate,
    create_array,
    create_group_with_attrs,
    open_group,
    write_region,
)


class ArrayContinuousSource:
    """In-memory ContinuousChannelSource backed by a float32 array."""

    def __init__(
        self,
        samples,
        *,
        id="ch-0",
        rate_hz=32000.0,
        start_us=0,
        name="ch-0",
        unit="uV",
        provenance=None,
    ):
        self._samples = np.asarray(samples, dtype=np.float32)
        self.id = id
        self.name = name
        self.unit = unit
        self._rate_hz = rate_hz
        self._start_us = start_us
        self._provenance = provenance

    def provenance(self):
        return self._provenance

    def rate_hz(self):
        return self._rate_hz

    def start_us(self):
        return self._start_us

    def num_samples(self):
        return int(self._samples.shape[0])

    def read_samples(self, start, stop):
        return self._samples[start:stop]


@pytest.fixture
def continuous_source():
    """Return the ArrayContinuousSource factory."""
    return ArrayContinuousSource


class RunsContinuousSource(ArrayContinuousSource):
    """In-memory continuous source that states its runs.

    The source holds only recorded samples. Each run is a pair of the time of
    its first sample in microseconds since the Unix epoch and its samples.
    start_us is the time of the first run.
    """

    def __init__(self, runs, **options):
        runs = [(start, np.asarray(samples)) for start, samples in runs]
        options["start_us"] = runs[0][0]
        super().__init__(
            np.concatenate([samples for _, samples in runs]), **options
        )
        self._runs = [
            Run(start, int(samples.shape[0])) for start, samples in runs
        ]

    def runs(self):
        return self._runs


@pytest.fixture
def runs_source():
    """Return the RunsContinuousSource factory."""
    return RunsContinuousSource


class ArrayMultiChannelSource:
    """In-memory MultiChannelSource over ArrayContinuousSource channels.

    Each window holds window_rows samples of every channel. window_rows is one
    int for all channels or one int per channel. A channel at its end gives an
    empty block, and the windows stop when every channel is at its end.
    """

    def __init__(self, sources, window_rows=8):
        self._sources = list(sources)
        if isinstance(window_rows, int):
            window_rows = [window_rows] * len(self._sources)
        self._window_rows = list(window_rows)

    def channels(self):
        return self._sources

    def iter_windows(self):
        positions = [0] * len(self._sources)
        while any(
            position < source.num_samples()
            for position, source in zip(positions, self._sources, strict=True)
        ):
            window = []
            for index, source in enumerate(self._sources):
                start = positions[index]
                stop = min(
                    start + self._window_rows[index], source.num_samples()
                )
                window.append(source.read_samples(start, stop))
                positions[index] = stop
            yield window


@pytest.fixture
def multi_channel_source():
    """Return the ArrayMultiChannelSource factory."""
    return ArrayMultiChannelSource


class ArrayEventSource:
    """In-memory EventChannelSource backed by arrays.

    events are int64 absolute microseconds. An optional column is offered only
    when its data is given. The label count, the text width, and the duration
    bound come from the data unless passed. A source without events and with
    labels has 1 label category.
    """

    def __init__(
        self,
        events,
        *,
        durations=None,
        labels=None,
        label_names=None,
        text=None,
        channel_ids=None,
        waveform_mean=None,
        waveform_period_us=100.0,
        waveform_event_sample=0,
        max_duration_us=None,
        start_us=None,
        id="evt-0",
        name="evt-0",
        unit=None,
        provenance=None,
    ):
        self._events = np.asarray(events, dtype=np.int64)
        self._durations = (
            None if durations is None else np.asarray(durations, dtype=np.int64)
        )
        self._labels = (
            None if labels is None else np.asarray(labels, dtype=np.uint16)
        )
        self._label_names = label_names
        self._text = None if text is None else list(text)
        self._channel_ids = (
            None if channel_ids is None else [list(ids) for ids in channel_ids]
        )
        self._waveform_mean = (
            None
            if waveform_mean is None
            else np.asarray(waveform_mean, dtype=np.float32)
        )
        self._waveform_period_us = waveform_period_us
        self._waveform_event_sample = waveform_event_sample
        self._max_duration_us = max_duration_us
        self._start_us = start_us
        self.id = id
        self.name = name
        self.unit = unit
        self._provenance = provenance

    def provenance(self):
        return self._provenance

    def start_us(self):
        if self._start_us is not None:
            return self._start_us
        return int(self._events[0]) if self._events.shape[0] else 0

    def num_events(self):
        return int(self._events.shape[0])

    def columns(self):
        present = {
            "durations": self._durations,
            "labels": self._labels,
            "text": self._text,
            "channels": self._channel_ids,
            "waveform_mean": self._waveform_mean,
        }
        return frozenset(
            key for key, value in present.items() if value is not None
        )

    def label_names(self):
        return self._label_names

    def num_labels(self):
        if self._labels is None:
            return 0
        if self._label_names is not None:
            return len(self._label_names)
        return int(self._labels.max()) + 1 if self._labels.shape[0] else 1

    def max_duration_us(self):
        if self._durations is None:
            return None
        if self._max_duration_us is not None:
            return self._max_duration_us
        return int(self._durations.max(initial=0))

    def text_width(self):
        return 0 if self._text is None else text_width(self._text)

    def read_events(self, start, stop):
        return self._events[start:stop]

    def read_durations(self, start, stop):
        return self._durations[start:stop]

    def read_labels(self, start, stop):
        return self._labels[start:stop]

    def read_text(self, start, stop):
        return self._text[start:stop]

    def read_channel_ids(self, start, stop):
        return self._channel_ids[start:stop]

    def waveform_mean(self):
        if self._waveform_mean is None:
            raise ValueError("this source has no waveform_mean column")
        return self._waveform_mean

    def waveform_period_us(self):
        return self._waveform_period_us

    def waveform_event_sample(self):
        return self._waveform_event_sample


@pytest.fixture
def event_source():
    """Return the ArrayEventSource factory."""
    return ArrayEventSource


def write_mixed_bundle(
    path,
    *,
    recording=_RECORDING,
    base=_SESSION_US,
    opts=_SPEC_OPTS,
    with_provenance=True,
):
    """Write the mixed bundle of tests/test_bundle.py at path and return path.

    It holds continuous channels 0 and 1 and event channels 2, 3, and 4. base
    is the start of channel 0 in microseconds since the Unix epoch.
    """
    continuous, events = _mixed_sources(
        ArrayContinuousSource, ArrayEventSource, base, with_provenance
    )
    write_bundle(
        continuous, events, output_path=path, opts=opts, recording=recording
    )
    return path


PAIR_LENGTHS = (4096, 3000)
"""Sample counts of the two channels of the pair bundle."""


def write_pair_bundle(path):
    """Write a bundle of two 512 Hz channels that start together at path.

    Channel 0 holds more samples than channel 1, and level 1 of each holds
    minmax, mean, and count. Returns path.
    """
    sources = [
        ArrayContinuousSource(
            np.arange(length, dtype=np.float32),
            id=f"pair-{index}",
            name=f"pair-{index}",
            rate_hz=512.0,
            start_us=10_000_000,
        )
        for index, length in enumerate(PAIR_LENGTHS)
    ]
    write_bundle(sources, [], output_path=path, opts=_SPEC_OPTS)
    return path


RUNS_START_US = 10_000_000
"""Time of the first sample of channel 0 of the runs bundle, as epoch microseconds."""
RUNS_RATE_HZ = 512.0
RUNS_LENGTHS = (4096, 2048)
"""Sample counts of the two runs of channel 0 of the runs bundle."""
RUNS_RESUME_US = 9_500_700
"""Time of the first sample of the second run, from the first sample.

The first run ends at 8 000 000 us. The resume is 1 500 700 us later, which is
0.358 of a sample period past grid sample 4864.
"""
RUNS_TABLE = ((0, 4096, 0), (4864, 2048, RUNS_RESUME_US))
"""The runs array of channel 0 of the runs bundle."""
RUNS_GRID_LENGTH = 6912
"""Length of raw in both channels of the runs bundle."""


def runs_samples():
    """Return the recorded samples of channel 0 of the runs bundle, per run."""
    rng = np.random.default_rng(11)
    return [
        (100.0 * (index + 1) + 8.0 * rng.standard_normal(length)).astype(
            np.float32
        )
        for index, length in enumerate(RUNS_LENGTHS)
    ]


def write_runs_bundle(path):
    """Write a bundle whose channel 0 resumes off its sample grid at path.

    Channel 0 holds two runs at 512 Hz with a pause that ends between two grid
    samples, so its runs array is RUNS_TABLE and its raw array is NaN from
    sample 4096 through 4863. Channel 1 has no runs and holds RUNS_GRID_LENGTH
    samples that start with channel 0. The reference time is the first sample.
    Returns path.
    """
    first, second = runs_samples()
    gapped = RunsContinuousSource(
        [
            (RUNS_START_US, first),
            (RUNS_START_US + RUNS_RESUME_US, second),
        ],
        id="runs-0",
        name="runs-0",
        rate_hz=RUNS_RATE_HZ,
    )
    plain = ArrayContinuousSource(
        np.arange(RUNS_GRID_LENGTH, dtype=np.float32),
        id="runs-1",
        name="runs-1",
        rate_hz=RUNS_RATE_HZ,
        start_us=RUNS_START_US,
    )
    write_bundle([gapped, plain], [], output_path=path, opts=_SPEC_OPTS)
    return path


MEMORY_PREFIX = "sub/bundle"
"""Prefix of the second copy of the mixed bundle in memory_bundle."""


def memory_store(path, prefixes=("",), omit=()):
    """Return a read-only MemoryStore holding every file under path.

    The store holds each file once under each prefix. The empty prefix is the
    store root. A file whose path below path is in omit stays out.
    """
    buffer = default_buffer_prototype().buffer
    files = {}
    for file in sorted(path.rglob("*")):
        name = file.relative_to(path).as_posix()
        if not file.is_file() or name in omit:
            continue
        data = buffer.from_bytes(file.read_bytes())
        for prefix in prefixes:
            files[f"{prefix}/{name}" if prefix else name] = data
    return MemoryStore(files, read_only=True)


ODD_VERSION = "1.3"
ODD_ROOT_TSZ = {"version": ODD_VERSION, "future_key": "not in 1.0"}
ODD_KEYS = (0, 2, 5, 7, 9, 12)
ODD_INNER_ROWS = 256
ODD_SHARD_ROWS = 1024
ODD_ZSTD_LEVEL = 3
ODD_RATE_HZ = 512.0
ODD_NAN_RUN = (100, 140)
ODD_MEDIAN_KEY = "median"
ODD_OFFSET_US = 2_000_000
ODD_EVENTS = 64
ODD_LABELS = 3
ODD_TIE = 10
"""Index of the event whose time the next event repeats in channel 9."""
ODD_COUNT_PERIOD_US = 1_024_000
ODD_CHANNEL_SETS = ((0,), (2, 5), ())
ODD_TEXT_WIDTH = 5
ODD_FULL_TEXT = "caf\N{LATIN SMALL LETTER E WITH ACUTE}"
ODD_FULL_TEXT_ROW = 20
ODD_WAVEFORM_SAMPLES = 8
ODD_WAVEFORM_PERIOD_US = 1e6 / 30_000
ODD_EVENT_SAMPLE = 3
ODD_INTERVAL_EVENTS = (500_000, 1_500_000, 4_000_000, 6_000_000, 9_000_000)
ODD_INTERVAL_DURATIONS = (2_000_000, 0, 6_000_000, 100_000, 0)


def _odd_signal(num_samples, seed):
    """Return num_samples float32 samples of seeded noise."""
    rng = np.random.default_rng(seed)
    return (100.0 * rng.standard_normal(num_samples)).astype(np.float32)


def _write_rows(group, name, data, dimension_names=None, attrs=None):
    """Create the array name under group and write data one shard at a time.

    Inner chunks hold ODD_INNER_ROWS rows and shards ODD_SHARD_ROWS rows.
    Every trailing axis is whole.
    """
    trailing = data.shape[1:]
    array = create_array(
        group,
        name,
        data.shape,
        data.dtype,
        (ODD_INNER_ROWS, *trailing),
        (ODD_SHARD_ROWS, *trailing),
        {} if attrs is None else attrs,
        ODD_ZSTD_LEVEL,
        dimension_names=dimension_names,
    )
    for start in range(0, data.shape[0], ODD_SHARD_ROWS):
        write_region(array, start, data[start : start + ODD_SHARD_ROWS])


def _reduce_bins(values, width, reduce):
    """Apply reduce to each bin of width values, the short last bin included."""
    whole = values.shape[0] - values.shape[0] % width
    parts = [reduce(values[:whole].reshape(-1, width), axis=1)]
    if whole < values.shape[0]:
        parts.append(reduce(values[whole:].reshape(1, -1), axis=1))
    return np.concatenate(parts)


def _level_statistics(samples, samples_per_bin):
    """Return every member of a level of samples_per_bin, keyed by name.

    Each member covers the finite samples of its bin. A bin with no finite
    sample has count 0 and NaN in every other member.
    """
    width = samples_per_bin
    values = samples.astype(np.float64)
    values[~np.isfinite(values)] = np.nan
    finite = _reduce_bins(np.isfinite(samples), width, np.sum)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        minimum = _reduce_bins(values, width, np.nanmin)
        maximum = _reduce_bins(values, width, np.nanmax)
        mean = _reduce_bins(values, width, np.nanmean)
        variance = _reduce_bins(values, width, np.nanvar)
        median = _reduce_bins(values, width, np.nanmedian)
    return {
        MINMAX_KEY: np.stack([minimum, maximum], axis=1).astype(np.float32),
        MEAN_KEY: mean.astype(np.float32),
        VAR_KEY: variance.astype(np.float32),
        COUNT_KEY: finite.astype(np.uint32),
        ODD_MEDIAN_KEY: median.astype(np.float32),
    }


def _write_continuous(root, key, attrs, samples, levels, widths=None):
    """Write continuous channel key with raw and the named level members.

    levels maps each level number to the names of the members it holds. widths
    lists the samples per bin of level 1, 2, and so on, and the default is
    DEFAULT_DECIMATION_FACTOR to the power of the level number.
    """
    channel = create_group_with_attrs(root, str(key), attrs)
    _write_rows(channel, RAW_KEY, samples, (TIME_DIMENSION,))
    rate_hz = attrs[TSZ_KEY]["rate_hz"]
    for level, members in levels.items():
        width = (
            DEFAULT_DECIMATION_FACTOR**level
            if widths is None
            else widths[level - 1]
        )
        statistics = _level_statistics(samples, width)
        group = create_group_with_attrs(
            channel,
            str(level),
            level_group_attrs(level_period_us(rate_hz, width)),
        )
        for member in members:
            dimensions = (
                (TIME_DIMENSION, None)
                if member == MINMAX_KEY
                else (TIME_DIMENSION,)
            )
            _write_rows(group, member, statistics[member], dimensions)


def _write_marks(root):
    """Write event channel 9 with every per-event column but durations."""
    channel = create_group_with_attrs(
        root, "9", {TSZ_KEY: {"kind": "event", "name": "marks", "unit": "a.u."}}
    )
    events = 1_000_000 + 250_000 * np.arange(ODD_EVENTS, dtype=np.int64)
    events[ODD_TIE + 1] = events[ODD_TIE]
    labels = (np.arange(ODD_EVENTS) % ODD_LABELS).astype(np.uint16)
    texts = [f"e{row}" if row % 4 == 1 else "" for row in range(ODD_EVENTS)]
    texts[ODD_FULL_TEXT_ROW] = ODD_FULL_TEXT
    key_lists = [
        ODD_CHANNEL_SETS[row % len(ODD_CHANNEL_SETS)]
        for row in range(ODD_EVENTS)
    ]
    waveforms = (
        np.random.default_rng(9)
        .standard_normal((ODD_EVENTS, ODD_WAVEFORM_SAMPLES))
        .astype(np.float32)
    )
    waveform_mean = np.stack(
        [
            waveforms[labels == label].astype(np.float64).mean(axis=0)
            for label in range(ODD_LABELS)
        ]
    ).astype(np.float32)
    counts = np.zeros(
        (int(events[-1]) // ODD_COUNT_PERIOD_US + 1, ODD_LABELS),
        dtype=np.uint32,
    )
    np.add.at(counts, (events // ODD_COUNT_PERIOD_US, labels), 1)
    snippet_attrs = waveform_array_attrs(
        ODD_WAVEFORM_PERIOD_US, ODD_EVENT_SAMPLE
    )
    event_dimensions = (EVENT_DIMENSION,)
    row_dimensions = (EVENT_DIMENSION, None)
    _write_rows(channel, EVENTS_KEY, events, event_dimensions)
    _write_rows(channel, LABELS_KEY, labels, event_dimensions)
    values = np.arange(ODD_EVENTS, dtype=np.float32) / 8
    _write_rows(channel, VALUES_KEY, values, event_dimensions)
    text = text_rows(texts, ODD_TEXT_WIDTH)
    _write_rows(channel, TEXT_KEY, text, row_dimensions)
    bitmasks = channels_rows(key_lists, max(ODD_KEYS))
    _write_rows(channel, CHANNELS_KEY, bitmasks, row_dimensions)
    _write_rows(
        channel, WAVEFORMS_KEY, waveforms, row_dimensions, snippet_attrs
    )
    label_dimensions = (LABEL_DIMENSION, None)
    _write_rows(
        channel,
        WAVEFORM_MEAN_KEY,
        waveform_mean,
        label_dimensions,
        snippet_attrs,
    )
    level = create_group_with_attrs(
        channel, "1", {TSZ_KEY: {"period_us": ODD_COUNT_PERIOD_US}}
    )
    _write_rows(level, COUNT_KEY, counts, (TIME_DIMENSION, None))
    create_group_with_attrs(channel, "notes", {})


def _write_intervals(root):
    """Write event channel 12, which holds durations and no bound on them."""
    channel = create_group_with_attrs(
        root, "12", {TSZ_KEY: {"kind": "event", "name": "intervals"}}
    )
    event_dimensions = (EVENT_DIMENSION,)
    events = np.array(ODD_INTERVAL_EVENTS, dtype=np.int64)
    durations = np.array(ODD_INTERVAL_DURATIONS, dtype=np.int64)
    _write_rows(channel, EVENTS_KEY, events, event_dimensions)
    _write_rows(channel, DURATIONS_KEY, durations, event_dimensions)


def write_odd_bundle(path):
    """Write a bundle of edge cases that a 1.x reader opens, and return path.

    The root tsz holds version ODD_VERSION and a key 1.0 does not define. The
    root also holds the array index and the group montages, which are not
    channels. There is no recording.json. Every array has inner chunks of
    ODD_INNER_ROWS rows and shards of ODD_SHARD_ROWS rows.

    Channel 0 is continuous at ODD_RATE_HZ, omits id and offset_us, and holds
    1001 samples. The samples in ODD_NAN_RUN are NaN. Level 1 holds minmax
    and an unknown median array. Level 2 holds minmax and count. The last bin
    of each level is short. Channel 2 is continuous at ODD_RATE_HZ with id two
    and 3000 samples. Its level 1 holds minmax, mean, var, and count. Channel
    5 has 1024 samples from ODD_OFFSET_US and no levels. Channel 7 is
    continuous at 30000 Hz with no samples. Channel 9 is an event channel that
    omits id and label_names. It holds ODD_EVENTS events with one tie, every
    per-event column but durations, count level 1, and an unknown group notes.
    Channel 12 holds durations and omits max_duration_us.
    """
    root = open_group(path, {TSZ_KEY: dict(ODD_ROOT_TSZ)})
    _write_rows(root, "index", np.array(ODD_KEYS, dtype=np.int64))
    create_group_with_attrs(root, "montages", {"montage": {"pairs": [[0, 2]]}})
    short = _odd_signal(1001, 0)
    short[slice(*ODD_NAN_RUN)] = np.nan
    _write_continuous(
        root,
        0,
        {
            TSZ_KEY: {
                "kind": "continuous",
                "name": "short",
                "rate_hz": ODD_RATE_HZ,
                "unit": "uV",
            }
        },
        short,
        {1: (MINMAX_KEY, ODD_MEDIAN_KEY), 2: (MINMAX_KEY, COUNT_KEY)},
    )
    _write_continuous(
        root,
        2,
        continuous_channel_attrs("full", "two", ODD_RATE_HZ, 0, "uV", None),
        _odd_signal(3000, 2),
        {1: (MINMAX_KEY, MEAN_KEY, VAR_KEY, COUNT_KEY)},
    )
    _write_continuous(
        root,
        5,
        continuous_channel_attrs(
            "late", "five", ODD_RATE_HZ, ODD_OFFSET_US, "uV", None
        ),
        _odd_signal(1024, 5),
        {},
    )
    _write_continuous(
        root,
        7,
        continuous_channel_attrs("empty", "seven", 30_000.0, 0, "uV", None),
        np.zeros(0, dtype=np.float32),
        {},
    )
    _write_marks(root)
    _write_intervals(root)
    consolidate(root)
    return path


FULL_RATE_HZ = 512.0
FULL_SAMPLES = 1001
FULL_GAP = (400, 500)
"""Samples of channel 0 of the full bundle that lie outside every run."""
FULL_RESUME_US = 976_563
"""Time of the first sample of the second run, 0.5 microsecond past its grid time."""
FULL_EVENTS = 40
FULL_LABELS = 3
FULL_WAVEFORM_SAMPLES = 8
FULL_COUNT_PERIOD_US = 1_024_000
FULL_MAX_DURATION_US = 6_000_000
FULL_TEXT_WIDTH = 6


def _full_continuous(root):
    """Write channel 0 of the full bundle: two runs, two levels, a gap."""
    samples = _odd_signal(FULL_SAMPLES, 4)
    samples[10] = np.inf
    samples[20] = np.nan
    samples[slice(*FULL_GAP)] = np.nan
    _write_continuous(
        root,
        0,
        continuous_channel_attrs(
            "runs", "zero", FULL_RATE_HZ, 0, "uV", {"software": "test"}
        ),
        samples,
        {
            1: (MINMAX_KEY, MEAN_KEY, VAR_KEY, COUNT_KEY),
            2: (MINMAX_KEY, COUNT_KEY, MEAN_KEY),
        },
    )
    runs = np.array(
        [
            [0, FULL_GAP[0], 0],
            [FULL_GAP[1], FULL_SAMPLES - FULL_GAP[1], FULL_RESUME_US],
        ],
        dtype=np.int64,
    )
    _write_rows(root["0"], RUNS_KEY, runs, ("run", None))
    _write_continuous(
        root,
        1,
        continuous_channel_attrs("empty", "one", 30_000.0, 0, "uV", None),
        np.zeros(0, dtype=np.float32),
        {},
    )


def _full_waveforms(labels):
    """Return waveforms with NaN values, and the mean of each label's rows."""
    rng = np.random.default_rng(21)
    waveforms = rng.standard_normal((FULL_EVENTS, FULL_WAVEFORM_SAMPLES))
    waveforms = waveforms.astype(np.float32)
    waveforms[3, 2] = np.nan
    waveforms[labels == 1, 5] = np.nan
    means = np.full((FULL_LABELS, FULL_WAVEFORM_SAMPLES), np.nan)
    for label in range(FULL_LABELS):
        rows = waveforms[labels == label].astype(np.float64)
        for sample in range(FULL_WAVEFORM_SAMPLES):
            finite = rows[np.isfinite(rows[:, sample]), sample]
            if finite.size:
                means[label, sample] = finite.mean()
    return waveforms, means.astype(np.float32)


def _full_events(root):
    """Write channel 2 of the full bundle: every column and two count levels."""
    spare = np.arange(FULL_EVENTS)
    events = 1_000_000 + 250_000 * spare.astype(np.int64)
    events[ODD_TIE + 1] = events[ODD_TIE]
    labels = (spare % FULL_LABELS).astype(np.uint16)
    durations = np.where(spare % 5 == 0, 1_500_000, 0).astype(np.int64)
    texts = ["", ODD_FULL_TEXT, "note", "abcdef"] * (FULL_EVENTS // 4)
    keys = [(0,), (1, 2), (), (0, 1, 2)] * (FULL_EVENTS // 4)
    waveforms, waveform_mean = _full_waveforms(labels)
    snippet_attrs = waveform_array_attrs(
        ODD_WAVEFORM_PERIOD_US, ODD_EVENT_SAMPLE, FULL_WAVEFORM_SAMPLES
    )
    channel = create_group_with_attrs(
        root,
        "2",
        event_channel_attrs(
            "events",
            "two",
            "uV",
            ["a", "b", "c"],
            FULL_MAX_DURATION_US,
            None,
        ),
    )
    event_dimensions = (EVENT_DIMENSION,)
    row_dimensions = (EVENT_DIMENSION, None)
    _write_rows(channel, EVENTS_KEY, events, event_dimensions)
    _write_rows(channel, DURATIONS_KEY, durations, event_dimensions)
    _write_rows(channel, LABELS_KEY, labels, event_dimensions)
    values = (np.arange(FULL_EVENTS) / 8).astype(np.float32)
    _write_rows(channel, VALUES_KEY, values, event_dimensions)
    _write_rows(
        channel, TEXT_KEY, text_rows(texts, FULL_TEXT_WIDTH), row_dimensions
    )
    _write_rows(channel, CHANNELS_KEY, channels_rows(keys, 2), row_dimensions)
    _write_rows(
        channel, WAVEFORMS_KEY, waveforms, row_dimensions, snippet_attrs
    )
    _write_rows(
        channel,
        WAVEFORM_MEAN_KEY,
        waveform_mean,
        (LABEL_DIMENSION, None),
        snippet_attrs,
    )
    for level, factor in ((1, 1), (2, 4)):
        period = FULL_COUNT_PERIOD_US * factor
        group = create_group_with_attrs(
            channel, str(level), level_group_attrs(period, 0)
        )
        counts = np.zeros(
            (int(events[-1]) // period + 1, FULL_LABELS), dtype=np.uint32
        )
        np.add.at(counts, (events // period, labels), 1)
        _write_rows(group, COUNT_KEY, counts, (TIME_DIMENSION, None))


def write_full_bundle(path):
    """Write a bundle that holds every array of the specification at path.

    Channel 0 is continuous at FULL_RATE_HZ with two runs, a gap of NaN
    samples between them, an infinite sample, and levels 1 and 2. Channel 1
    is continuous and empty. Channel 2 is an event channel with FULL_EVENTS
    events, every per-event column, waveforms, waveform_mean, and two count
    levels. Event channel 2 sits beside two continuous channels, so channels
    holds one byte per row. recording.json holds a reference time. Returns
    path.
    """
    root = open_group(path, root_attrs())
    _full_continuous(root)
    _full_events(root)
    consolidate(root)
    (path / "recording.json").write_text(
        json.dumps({"reference_time": "2024-03-14T14:15:00+00:00"})
    )
    return path


@pytest.fixture(scope="session")
def full_bundle(tmp_path_factory):
    """Return the bundle write_full_bundle writes, written once per session."""
    return write_full_bundle(tmp_path_factory.mktemp("full") / "bundle")


@pytest.fixture(scope="session")
def mixed_bundle(tmp_path_factory):
    """Return the default mixed bundle, written once per session.

    Tests only read it.
    """
    return write_mixed_bundle(tmp_path_factory.mktemp("mixed") / "bundle")


@pytest.fixture
def write_mixed(tmp_path):
    """Return a factory that writes a mixed bundle variant under tmp_path.

    The factory takes the directory name, then the keyword arguments of
    write_mixed_bundle, and returns the bundle path.
    """

    def write(name="bundle", **options):
        return write_mixed_bundle(tmp_path / name, **options)

    return write


@pytest.fixture(scope="session")
def memory_bundle(mixed_bundle):
    """Return a read-only MemoryStore holding the mixed bundle twice.

    One copy sits at the store root and one under MEMORY_PREFIX.
    """
    return memory_store(mixed_bundle, ("", MEMORY_PREFIX))


@pytest.fixture(scope="session")
def runs_bundle(tmp_path_factory):
    """Return the bundle write_runs_bundle writes, written once per session."""
    return write_runs_bundle(tmp_path_factory.mktemp("runs") / "bundle")


@pytest.fixture(scope="session")
def odd_bundle(tmp_path_factory):
    """Return the bundle write_odd_bundle writes, written once per session."""
    return write_odd_bundle(tmp_path_factory.mktemp("odd") / "bundle")
