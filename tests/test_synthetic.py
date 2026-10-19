"""Reproduce the notebook demo_reading_with_zarr.ipynb through open_bundle."""

import json

import numpy as np
import pytest

from timeseries_zarr.read_bundle import open_bundle
from timeseries_zarr.read_continuous import Bins, Samples

_PIXELS = 1200
_DC_OFFSET_UV = 5000.0


@pytest.fixture(scope="module")
def bundle(synthetic_bundle):
    return open_bundle(synthetic_bundle)


def test_discovery_lists_every_channel_from_the_root_metadata(bundle):
    assert bundle.version == "1.0"
    assert bundle.describe().splitlines() == [
        "key  kind        name              id              unit  rate_hz  "
        "length          levels",
        "0    continuous  EEG-like          synth-eeg-01    uV    1024.0   "
        "614400 samples  1-5",
        "1    continuous  slow-wave+offset  synth-slow-02   uV    256.0    "
        "153600 samples  1-4",
        "2    continuous  gappy             synth-gap-03    uV    512.0    "
        "307200 samples  1-5",
        "3    event       units             synth-units-04  -     -        "
        "8099 events     1-2",
        "4    event       curated marks     synth-marks-05  -     -        "
        "14 events       -",
        "5    event       hfo detector      synth-det-06    a.u.  -        "
        "5000 events     1-2",
    ]


def test_discovery_lists_the_period_of_every_level(bundle):
    continuous = {
        key: {
            number: level.period_us for number, level in channel.levels.items()
        }
        for key, channel in bundle.continuous.items()
    }
    assert continuous == {
        0: {1: 3906.25, 2: 15625.0, 3: 62500.0, 4: 250000.0, 5: 1e6},
        1: {1: 15625.0, 2: 62500.0, 3: 250000.0, 4: 1e6},
        2: {1: 7812.5, 2: 31250.0, 3: 125000.0, 4: 500000.0, 5: 2e6},
    }
    events = {
        key: {
            number: level.period_us
            for number, level in channel.count_levels.items()
        }
        for key, channel in bundle.events.items()
    }
    assert events == {
        3: {1: 256000.0, 2: 1.024e6},
        4: {},
        5: {1: 256000.0, 2: 1.024e6},
    }


def test_discovery_lists_the_columns_of_each_event_channel(bundle):
    columns = {key: channel.columns for key, channel in bundle.events.items()}
    assert columns == {
        3: ("labels", "waveforms"),
        4: ("durations", "labels", "text", "channels"),
        5: ("labels", "values", "text", "channels"),
    }
    assert bundle.events[3].waveform_mean is not None
    assert bundle.events[4].waveform_mean is None


def test_recording_reads_the_identity_and_no_time_from_a_legacy_start_key(
    bundle,
):
    recording = bundle.recording
    assert recording.subject["id"] == "sub-synth01"
    assert recording.session["id"] == "ses-01"
    assert recording.source["seed"] == 2026
    assert recording.reference_time is None
    assert recording.session_start_time is None


def test_first_mark_is_37_54_seconds_after_the_reference_time(bundle):
    assert bundle.events[4].read().time_us[0] == 37_540_283


@pytest.mark.parametrize(
    ("start_us", "end_us", "level", "first", "length"),
    [
        (0, 600_000_000, 4, 0, 2400),
        (300_000_000, 330_000_000, 2, 19200, 1920),
        (312_000_000, 314_000_000, 0, 319488, 2048),
    ],
    ids=["0-600 s", "300-330 s", "312-314 s"],
)
def test_zoom_window_reads_the_coarsest_level_that_fills_1200_bins(
    bundle, start_us, end_us, level, first, length
):
    channel = bundle.continuous[0]
    chosen = channel.level_for(start_us, end_us, bins=_PIXELS)
    assert chosen == level
    window = channel.read(start_us, end_us, level=chosen)
    assert isinstance(window, Bins if level else Samples)
    assert (window.first, len(window)) == (first, length)
    assert window.time_us[0] == start_us


def test_raw_zoom_window_starts_at_312_seconds(bundle):
    samples = bundle.continuous[0].read(312_000_000, 314_000_000)
    assert samples.time_s[0] == 312.0
    assert samples.time_s[-1] == 312.0 + 2047 / 1024


def test_offset_channel_moves_mean_and_leaves_var(bundle):
    channel = bundle.continuous[1]
    assert channel.offset_us == 500_000
    bins = channel.read(level=3)
    assert len(bins) == 2400
    assert bins.period_us == 250_000.0
    assert bins.time_us[:2].tolist() == [500_000.0, 750_000.0]
    mean = bins.mean.astype(np.float64)
    assert mean.min() == pytest.approx(4946.4, abs=0.05)
    assert mean.max() == pytest.approx(5060.5, abs=0.05)
    raw = channel.read().values.astype(np.float64) - _DC_OFFSET_UV
    width = 4**3
    direct = np.array(
        [
            raw[start : start + width].var()
            for start in range(0, raw.size, width)
        ]
    )
    relative = np.abs(bins.var.astype(np.float64) - direct) / direct
    assert relative.max() < 1e-6


def test_gappy_channel_coverage_comes_from_the_coarsest_count(bundle):
    channel = bundle.continuous[2]
    coarsest = max(channel.levels)
    bins = channel.read(level=coarsest)
    assert (coarsest, len(bins), bins.period_us) == (5, 300, 2e6)
    width = np.minimum(
        4**coarsest,
        channel.num_samples - np.arange(len(bins)) * 4**coarsest,
    )
    short = np.flatnonzero(bins.count < width)
    assert short.tolist() == [60, 61, 62, 125, 200]
    assert bins.count[short].tolist() == [0, 0, 768, 1023, 614]
    assert np.isnan(bins.min[short]).all()


def test_gappy_channel_bin_short_by_one_holds_one_nan_sample(bundle):
    samples = bundle.continuous[2].read(250_000_000, 252_000_000)
    assert (samples.first, len(samples)) == (128_000, 1024)
    assert np.flatnonzero(np.isnan(samples.values)).tolist() == [0]
    assert samples.time_s[0] == 250.0


def test_gappy_channel_spans_skip_the_dropouts(bundle):
    channel = bundle.continuous[2]
    assert channel.spans() == [(0, 120_000_000), (124_000_000, 600_000_000)]
    assert channel.spans(level=1) == [
        (0, 120_000_000),
        (124_500_000, 400_203_125),
        (401_000_000, 600_000_000),
    ]


def test_marks_from_100_to_300_seconds_are_the_seven_open_marks(bundle):
    table = bundle.events[4].read(100_000_000, 300_000_000)
    assert table.time_us.tolist() == [
        111_014_882,
        115_254_398,
        120_000_000,
        123_075_056,
        142_306_524,
        148_108_656,
        222_000_000,
    ]
    assert table.duration_us.tolist() == [0, 0, 4_500_000, 0, 0, 0, 90_000_000]
    assert table.label_name == [
        "spike",
        "spike",
        "note",
        "spike",
        "spike",
        "spike",
        "seizure",
    ]
    assert table.channel_keys == [[0], [0], [2], [0], [0], [0], []]
    spike = "interictal spike, reviewed"
    assert table.text == [
        spike,
        spike,
        "acquisition dropout",
        spike,
        spike,
        spike,
        "electrographic seizure, evolving 4 Hz",
    ]


def test_marks_that_apply_to_channel_0_leave_out_the_note_on_channel_2(bundle):
    table = bundle.events[4].read(100_000_000, 260_000_000)
    applies = table.applies_to(0)
    assert applies.tolist() == [True, True, False, True, True, True, True]
    skipped = np.flatnonzero(~applies)
    assert table.time_us[skipped].tolist() == [120_000_000]
    assert table.text[int(skipped[0])] == "acquisition dropout"


def test_detector_values_rank_the_events_and_text_holds_json(bundle):
    table = bundle.events[5].read()
    assert len(table) == 5000
    rows = [json.loads(text) for text in table.text]
    assert rows[0] == {"score": 0.3294, "band": "ripple"}
    top = np.argsort(table.value)[-3:][::-1]
    np.testing.assert_allclose(
        table.value[top], [0.94737792, 0.88927662, 0.84740794], rtol=1e-6
    )
    assert table.time_us[top].tolist() == [
        489_531_239,
        559_385_070,
        450_611_878,
    ]
    assert [table.text[index] for index in top.tolist()] == [
        '{"score":0.9474,"band":"ripple"}',
        '{"score":0.8893,"band":"fast-ripple"}',
        '{"score":0.8474,"band":"ripple"}',
    ]
    assert [rows[index]["band"] for index in top.tolist()] == [
        "ripple",
        "fast-ripple",
        "ripple",
    ]


def test_unit_raster_window_holds_194_events_of_five_units(bundle):
    channel = bundle.events[3]
    table = channel.read(300_000_000, 330_000_000)
    assert len(table) == 194
    per_label = np.bincount(table.label, minlength=len(channel.label_names))
    assert per_label.tolist() == [73, 9, 34, 49, 29]
    assert channel.label_names == tuple(f"unit-{index}" for index in range(5))
    assert table.waveform.values.shape == (194, 32)


def test_waveform_mean_axis_is_zero_at_the_event_sample(bundle):
    snippets = bundle.events[3].waveform_mean
    assert (snippets.event_sample, snippets.samples) == (12, 32)
    assert snippets.time_us[12] == 0.0
    np.testing.assert_array_equal(
        snippets.time_us,
        (np.arange(32) - snippets.event_sample) * snippets.period_us,
    )
    assert snippets.read().values.shape == (5, 32)


def test_count_level_1_summed_in_fours_equals_count_level_2(bundle):
    channel = bundle.events[3]
    level_1 = channel.read_counts(level=1)
    level_2 = channel.read_counts(level=2)
    assert level_2.count.shape == (586, 5)
    assert level_2.period_us == 1.024e6
    assert level_1.count.shape == (4 * 586, 5)
    np.testing.assert_array_equal(
        level_1.count.astype(np.int64).reshape(-1, 4, 5).sum(axis=1),
        level_2.count,
    )
    assert int(level_1.count.sum()) == int(level_2.count.sum()) == 8099
