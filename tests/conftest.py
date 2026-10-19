"""Shared pytest fixtures, the in-memory sources, and the test bundles."""

import os
from pathlib import Path

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
    level_group_attrs,
    waveform_array_attrs,
)
from timeseries_zarr.bundle import write_bundle
from timeseries_zarr.columns import channels_rows, text_rows, text_width
from timeseries_zarr.constants import (
    CHANNELS_KEY,
    COUNT_KEY,
    DECIMATION_FACTOR,
    DURATIONS_KEY,
    EVENT_DIMENSION,
    EVENTS_KEY,
    LABEL_DIMENSION,
    LABELS_KEY,
    MEAN_KEY,
    MINMAX_KEY,
    RAW_KEY,
    TEXT_KEY,
    TIME_DIMENSION,
    TSZ_KEY,
    VALUES_KEY,
    VAR_KEY,
    WAVEFORM_MEAN_KEY,
    WAVEFORMS_KEY,
)
from timeseries_zarr.planning import level_period_us
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


class ArrayEventSource:
    """In-memory EventChannelSource backed by arrays.

    events are int64 absolute microseconds. An optional column is offered only
    when its data is given. The label count, the text width, and the duration
    bound come from the data unless passed.
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
        return int(self._labels.max()) + 1 if self._labels.shape[0] else 0

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


MEMORY_PREFIX = "sub/bundle"
"""Prefix of the second copy of the mixed bundle in memory_bundle."""


def memory_store(path, prefixes=("",)):
    """Return a read-only MemoryStore holding every file under path.

    The store holds each file once under each prefix. The empty prefix is the
    store root.
    """
    buffer = default_buffer_prototype().buffer
    files = {}
    for file in sorted(path.rglob("*")):
        if not file.is_file():
            continue
        data = buffer.from_bytes(file.read_bytes())
        name = file.relative_to(path).as_posix()
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


def _write_rows(
    group, name, data, dimension_names=None, attrs=None, *, compress=True
):
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
        compress=compress,
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


def _level_statistics(samples, level):
    """Return every member of level folded from samples, keyed by name.

    A NaN sample makes minmax, mean, var, and median of its bin NaN. count
    holds the number of finite samples in each bin.
    """
    width = DECIMATION_FACTOR**level
    values = samples.astype(np.float64)
    minimum = _reduce_bins(values, width, np.min)
    maximum = _reduce_bins(values, width, np.max)
    finite = _reduce_bins(np.isfinite(samples), width, np.sum)
    return {
        MINMAX_KEY: np.stack([minimum, maximum], axis=1).astype(np.float32),
        MEAN_KEY: _reduce_bins(values, width, np.mean).astype(np.float32),
        VAR_KEY: _reduce_bins(values, width, np.var).astype(np.float32),
        COUNT_KEY: finite.astype(np.uint32),
        ODD_MEDIAN_KEY: _reduce_bins(values, width, np.median).astype(
            np.float32
        ),
    }


def _write_continuous(root, key, attrs, samples, levels):
    """Write continuous channel key with raw and the named level members.

    levels maps each level number to the names of the members it holds.
    """
    channel = create_group_with_attrs(root, str(key), attrs)
    _write_rows(channel, RAW_KEY, samples, (TIME_DIMENSION,))
    rate_hz = attrs[TSZ_KEY]["rate_hz"]
    for level, members in levels.items():
        statistics = _level_statistics(samples, level)
        group = create_group_with_attrs(
            channel,
            str(level),
            level_group_attrs(level_period_us(rate_hz, level)),
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
    _write_rows(channel, TEXT_KEY, text, row_dimensions, compress=False)
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
        channel, "1", level_group_attrs(float(ODD_COUNT_PERIOD_US))
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


SYNTHETIC_BUNDLE_ENV = "TSZ_SYNTHETIC_BUNDLE"
"""Environment variable naming the synthetic reference bundle."""

DEFAULT_SYNTHETIC_BUNDLE = (
    Path(__file__).resolve().parent.parent.parent
    / "timeseries-zarr-paper"
    / "testdata"
    / "synthetic.tszarr"
)
"""Where the paper repository beside this one keeps the synthetic bundle."""


def find_synthetic_bundle():
    """Return the synthetic reference bundle, or skip the test without one.

    SYNTHETIC_BUNDLE_ENV overrides DEFAULT_SYNTHETIC_BUNDLE when it is set
    and not empty.
    """
    given = os.environ.get(SYNTHETIC_BUNDLE_ENV)
    path = Path(given) if given else DEFAULT_SYNTHETIC_BUNDLE
    if not path.is_dir():
        pytest.skip(f"no synthetic bundle at {path}")
    return path


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
def odd_bundle(tmp_path_factory):
    """Return the bundle write_odd_bundle writes, written once per session."""
    return write_odd_bundle(tmp_path_factory.mktemp("odd") / "bundle")


@pytest.fixture(scope="session")
def synthetic_bundle():
    """Return the synthetic reference bundle, or skip without it."""
    return find_synthetic_bundle()
