"""Tests for the reading layer, validator, de-id tools, and CLI.

All tests run against the compact fixture bundle from
``tests.bundle_fixture`` (spec-complete: continuous, gappy,
viewing-only, misaligned, and event channels).
"""

import shutil

import numpy as np
import pytest
import zarr
from scipy import signal as sps

from tests.bundle_fixture import DUR_S, START_US, build_bundle
from timeseries_zarr import cli
from timeseries_zarr.deid import body_surfaces, dateshift, deidentify
from timeseries_zarr.reading import open_bundle
from timeseries_zarr.reading.filtering import decide_level
from timeseries_zarr.reading.montage import MontageAlignmentError
from timeseries_zarr.reading.serving import (
    FilterSpec,
    ServingError,
    Tier,
    classify,
)
from timeseries_zarr.validate import errors, validate_bundle


@pytest.fixture(scope="session")
def bundle_path(tmp_path_factory):
    return build_bundle(tmp_path_factory.mktemp("fix") / "b.tszarr")


@pytest.fixture(scope="session")
def bundle(bundle_path):
    return open_bundle(bundle_path)


# ------------------------------------------------------------- inventory
class TestBundle:
    def test_inventory(self, bundle):
        assert set(bundle.continuous) == {"0", "1", "2", "5", "6"}
        assert set(bundle.events) == {"3", "4"}
        ch0 = bundle["0"]
        assert ch0.rate_hz == 512.0
        assert [lv.k for lv in ch0.levels] == [1, 2, 3, 4]

    def test_meta_and_wall_clock(self, bundle):
        assert bundle.meta["subject"]["subject_id"] == "sub-fixture"
        assert bundle.start_us == START_US

    def test_viewing_only_flag(self, bundle):
        assert not bundle["6"].has_raw
        assert bundle["0"].has_raw


# --------------------------------------------------------------- windows
class TestWindows:
    def test_level_selection(self, bundle):
        w = bundle["0"].window(0, DUR_S, pixels=2000)
        # level bins over 120 s: k1=15360, k2=3840, k3=960, k4=240
        assert w.level == 2
        assert len(w.t) == pytest.approx(3840, abs=1)

    def test_env_contains_raw(self, bundle):
        ch = bundle["0"]
        w = ch.window(10, 20, pixels=500)
        raw = ch.raw_window(10, 20).raw
        assert w.env[:, 0].min() <= raw.min() + 1e-3
        assert w.env[:, 1].max() >= raw.max() - 1e-3

    def test_offset_uv_restored(self, bundle):
        w = bundle["1"].window(0, DUR_S, pixels=100)
        assert 900 < np.nanmean(w.mean) < 1100

    def test_short_window_serves_raw(self, bundle):
        w = bundle["0"].window(1.0, 1.05, pixels=200)
        assert w.level == 0
        assert w.raw is not None

    def test_viewing_only_serves_levels_but_not_raw(self, bundle):
        ch = bundle["6"]
        assert ch.window(0, DUR_S, pixels=1000).level > 0
        with pytest.raises(ServingError):
            ch.raw_window(0, 1)
        with pytest.raises(ServingError):
            ch.window(1.0, 1.05, pixels=200)

    def test_gap_propagates_to_mean(self, bundle):
        w = bundle["2"].window(38, 45, pixels=200)
        assert np.isnan(w.mean).any()


# --------------------------------------------------------------- montage
class TestMontage:
    def test_mean_is_exact(self, bundle):
        m = bundle.montage({"0": 1.0, "1": -1.0})
        w = m.window(0, DUR_S, pixels=2000)
        raw0 = bundle["0"].group["raw"][:].astype(np.float64)
        raw1 = bundle["1"].group["raw"][:].astype(np.float64)
        width = int(round(4**w.level))
        nb = len(w.t)
        truth = (raw0 - raw1)[: nb * width].reshape(nb, width).mean(1)
        scale = np.max(np.abs(truth))
        assert np.max(np.abs(w.mean - truth)) < 1e-3 * scale

    def test_band_contains_true_envelope(self, bundle):
        m = bundle.montage({"0": 1.0, "1": -1.0})
        w = m.window(0, DUR_S, pixels=500, with_band=True)
        raw0 = bundle["0"].group["raw"][:].astype(np.float64)
        raw1 = bundle["1"].group["raw"][:].astype(np.float64)
        width = int(round(4**w.level))
        nb = len(w.t)
        blocks = (raw0 - raw1)[: nb * width].reshape(nb, width)
        assert np.all(w.env[:, 0] <= blocks.min(1) + 1e-3)
        assert np.all(w.env[:, 1] >= blocks.max(1) - 1e-3)

    def test_cross_rate_refuses(self, bundle):
        with pytest.raises(MontageAlignmentError):
            bundle.montage({"0": 1.0, "2": -1.0})

    def test_misaligned_offset_refuses(self, bundle):
        m = bundle.montage({"0": 1.0, "5": -1.0})
        with pytest.raises(MontageAlignmentError):
            m.window(0, 30, pixels=500)


# ------------------------------------------------------------- filtering
class TestFiltering:
    def test_tier_classification(self):
        # 512 Hz native; level rates 128, 32, 8 Hz
        spec = FilterSpec(0.5, 4.0)
        assert classify(spec, 128.0)[1] is Tier.SILENT   # rho 0.0625
        assert classify(spec, 32.0)[1] is Tier.MARKED    # rho 0.25
        assert classify(spec, 8.0)[1] is Tier.REFUSE     # rho 1.0

    def test_highpass_quartered(self):
        spec = FilterSpec(0.5, None)
        rho, tier = classify(spec, 128.0)
        assert tier is Tier.SILENT and rho < 0.15 * 0.25
        # broadband halves on top of quartering
        _, tier_bb = classify(FilterSpec(2.0, None), 128.0,
                              broadband=True)
        assert tier_bb is Tier.MARKED  # rho 0.031 > 0.15*0.125

    def test_decide_picks_coarsest_admissible(self, bundle):
        spec = FilterSpec(0.5, 4.0)
        lv, decision = decide_level(
            bundle["0"], spec, 0, DUR_S, pixels=100
        )
        assert lv.k == 2 and decision.tier is Tier.MARKED

    def test_filtered_matches_native_ground_truth(self, bundle):
        ch = bundle["0"]
        w = ch.filtered_window(20, 100, low_hz=0.5, high_hz=4.0,
                               pixels=5000)
        assert w.level == 1 and not w.marked
        # ground truth: same causal design at native rate, from t=0,
        # then bin-averaged to the level grid
        raw = ch.group["raw"][:].astype(np.float64)
        sos = sps.butter(4, [0.5, 4.0], btype="bandpass",
                         fs=ch.rate_hz, output="sos")
        y = sps.sosfilt(sos, raw)
        nb = len(y) // 4
        truth_all = y[: nb * 4].reshape(nb, 4).mean(1)
        i0 = int(20 / w.period_s)
        truth = truth_all[i0:i0 + len(w.t)]
        rms = np.sqrt(np.mean((w.mean - truth) ** 2))
        scale = np.sqrt(np.mean(truth**2))
        assert rms / scale < 0.10

    def test_preroll_warms_up_a_seek(self, bundle):
        ch = bundle["0"]
        w = ch.filtered_window(60, 90, low_hz=0.5, high_hz=4.0,
                               pixels=2000)
        assert w.level == 1
        # reference: filter the whole level-1 series from t=0
        lv = ch.levels[0]
        full = ch.mean_series(lv, 0, lv.n_bins)
        sos = sps.butter(4, [0.5, 4.0], btype="bandpass",
                         fs=lv.rate_hz, output="sos")
        ref = sps.sosfilt(sos, full)
        i0 = int(round(60 / w.period_s))
        seg = ref[i0:i0 + len(w.t)]
        rms = np.sqrt(np.mean((w.mean - seg) ** 2))
        assert rms / np.sqrt(np.mean(seg**2)) < 0.05

    def test_refuse_falls_back_to_raw(self, bundle):
        w = bundle["0"].filtered_window(0, 10, low_hz=0.5,
                                        high_hz=70.0, pixels=100)
        assert w.level == 0 and w.marked
        assert "refused" in w.note

    def test_viewing_only_refuse_raises(self, bundle):
        with pytest.raises(ServingError):
            bundle["6"].filtered_window(0, 10, low_hz=0.5,
                                        high_hz=70.0, pixels=100)

    def test_gaps_blank_filtered_output(self, bundle):
        w = bundle["2"].filtered_window(30, 50, low_hz=0.5,
                                        high_hz=4.0, pixels=1000)
        assert np.isnan(w.mean).any()
        assert np.isfinite(w.mean[: len(w.mean) // 4]).all()


# ---------------------------------------------------------------- events
class TestEvents:
    def test_between(self, bundle):
        ev = bundle["3"].between(0, 60)
        assert list(ev["times_s"]) == [10.0, 30.0]
        assert ev["durations_s"][1] == 45.0

    def test_stabbing_query(self, bundle):
        # the 45 s seizure starts at t=30; a window at t=60 overlaps
        # it although its start lies before the window
        hits = bundle["3"].overlapping(60, 65)
        assert 30.0 in hits["times_s"]
        none = bundle["3"].overlapping(80, 90)
        assert len(none["times_s"]) == 0

    def test_bodies(self, bundle):
        assert bundle["3"].body(2) == "note"
        parsed = bundle["4"].body(0)
        assert isinstance(parsed, dict) and "score" in parsed

    def test_rates_are_exact(self, bundle):
        ch = bundle["4"]
        w = ch.rates(0, DUR_S, pixels=64)
        assert int(w.mean.sum()) == ch.n_events
        assert w.mean.shape[1] == 2


# --------------------------------------------------- validate, deid, cli
class TestValidateAndDeid:
    def test_fixture_is_conformant(self, bundle_path):
        findings = validate_bundle(bundle_path)
        assert errors(findings) == [], [str(f) for f in findings]

    def test_validator_catches_breakage(self, tmp_path):
        broken = tmp_path / "broken.tszarr"
        root = zarr.create_group(
            store=zarr.storage.LocalStore(broken), zarr_format=3
        )
        grp = root.create_group("0", attributes={
            "id": "x", "kind": "event", "name": "bad",
            "offset_us": 0})
        arr = grp.create_array("events", shape=(3,), dtype="int64")
        arr[:] = np.array([5, 2, 9])          # unsorted
        zarr.consolidate_metadata(root.store)
        findings = validate_bundle(broken)
        assert any("non-decreasing" in f.message
                   for f in errors(findings))

    def test_deid_and_dateshift(self, bundle_path, tmp_path):
        copy = tmp_path / "copy.tszarr"
        shutil.copytree(bundle_path, copy)
        new = dateshift(copy, int(3 * 86_400e6))
        assert new == START_US + int(3 * 86_400e6)
        surfaces = deidentify(copy)
        assert not (copy / "meta").exists()
        assert {s.channel for s in surfaces} == {"3", "4"}
        assert open_bundle(copy).meta is None
        assert errors(validate_bundle(copy)) == []
        with pytest.raises(FileNotFoundError):
            dateshift(copy, 1)

    def test_body_surfaces(self, bundle_path):
        surfaces = body_surfaces(bundle_path)
        json_ones = [s for s in surfaces
                     if s.media_type == "application/json"]
        assert json_ones and json_ones[0].n_bodies == 3000

    def test_cli_smoke(self, bundle_path, capsys):
        assert cli.main(["info", str(bundle_path)]) == 0
        out = capsys.readouterr().out
        assert "continuous" in out and "viewing-only" in out
        assert cli.main(["validate", str(bundle_path)]) == 0
