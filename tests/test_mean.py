"""The mean member and the DC offset that keeps it usable.

The claim under test is the one the mean member exists for: a fixed
re-reference computed from stored means equals the fold of the re-referenced
raw signal, at every level. If this ever fails, the pyramid stops being safe to
montage from and the viewer has to fetch raw.
"""

import numpy as np
import pytest

from timeseries_zarr.bundle import write_bundle
from timeseries_zarr.planning import plan_levels
from timeseries_zarr.types import WriteOpts
from timeseries_zarr.write_continuous import write_continuous_channel
from timeseries_zarr.zarr_io import open_group

_OPTS = WriteOpts(
    min_bins=2, max_levels=7, inner_len=16, target_shard_bytes=256
)

_RATE_HZ = 32000.0
_PERIOD_US = 31.25


def _levels(num_samples):
    return plan_levels(
        num_samples, _PERIOD_US, _OPTS.max_levels, _OPTS.min_bins
    )


def _write(tmp_path, sources):
    parent = open_group(tmp_path / "bundle")
    for index, source in enumerate(sources):
        write_continuous_channel(parent, index, source, onset_us=0, opts=_OPTS)
    return open_group(tmp_path / "bundle")


def _fold_mean(values, level):
    """Mean over disjoint blocks of 4**level, keeping a partial trailing bin."""
    span = 4**level
    return np.array(
        [values[i : i + span].mean() for i in range(0, values.shape[0], span)],
        dtype=np.float64,
    )


def test_mean_equals_the_direct_block_mean_at_every_level(
    tmp_path, continuous_source
):
    rng = np.random.default_rng(0)
    samples = rng.standard_normal(1024).astype(np.float32)
    root = _write(tmp_path, [continuous_source(samples, rate_hz=_RATE_HZ)])
    grp = root["0"]

    values = samples.astype(np.float64)
    for plan in _levels(samples.shape[0]):
        stored = grp[str(plan.level)]["mean"][:]
        assert np.allclose(stored, _fold_mean(values, plan.level), atol=1e-5)


def test_mean_is_exact_across_a_partial_trailing_bin(
    tmp_path, continuous_source
):
    """A length no power of 4 divides puts a short bin at the end of every level."""
    rng = np.random.default_rng(1)
    samples = rng.standard_normal(1013).astype(np.float32)
    root = _write(tmp_path, [continuous_source(samples, rate_hz=_RATE_HZ)])
    grp = root["0"]

    values = samples.astype(np.float64)
    for plan in _levels(samples.shape[0]):
        stored = grp[str(plan.level)]["mean"][:]
        expected = _fold_mean(values, plan.level)
        assert stored.shape == expected.shape
        # The last bin is short; an unweighted fold of means would miss here.
        assert np.allclose(stored, expected, atol=1e-5)


def test_montage_from_means_matches_the_montaged_signal(
    tmp_path, continuous_source
):
    """The paper's linear claim, asserted in the producer.

    A bipolar derivation taken from two channels' stored means equals folding
    the difference of their raw signals. Exact because the fold is linear and
    both channels share a bin grid.
    """
    rng = np.random.default_rng(2)
    a = rng.standard_normal(1024).astype(np.float32)
    b = rng.standard_normal(1024).astype(np.float32)
    root = _write(
        tmp_path,
        [
            continuous_source(a, id="a", rate_hz=_RATE_HZ),
            continuous_source(b, id="b", rate_hz=_RATE_HZ),
        ],
    )

    difference = a.astype(np.float64) - b.astype(np.float64)
    for plan in _levels(a.shape[0]):
        montage = root["0"][str(plan.level)]["mean"][:].astype(
            np.float64
        ) - root["1"][str(plan.level)]["mean"][:].astype(np.float64)
        assert np.allclose(
            montage, _fold_mean(difference, plan.level), atol=1e-4
        )


def test_mean_propagates_nan_rather_than_skipping_it(
    tmp_path, continuous_source
):
    """A NaN-aware mean would average different time supports across channels."""
    samples = np.ones(1024, dtype=np.float32)
    samples[5] = np.nan
    root = _write(tmp_path, [continuous_source(samples, rate_hz=_RATE_HZ)])
    grp = root["0"]

    level1 = grp["1"]["mean"][:]
    assert np.isnan(level1[1])
    assert level1[0] == 1.0
    # And it keeps propagating up rather than being averaged away.
    assert np.isnan(grp["2"]["mean"][0])


def test_offset_uv_is_written_only_when_there_is_one(
    tmp_path, continuous_source
):
    samples = np.arange(1024, dtype=np.float32)
    root = _write(
        tmp_path,
        [
            continuous_source(samples, id="plain", rate_hz=_RATE_HZ),
            continuous_source(
                samples, id="biased", rate_hz=_RATE_HZ, offset_uv=500_000.0
            ),
        ],
    )
    assert "offset_uv" not in dict(root["0"].attrs)
    assert dict(root["1"].attrs)["offset_uv"] == 500_000.0


def test_raw_keeps_its_offset_and_the_statistics_do_not(
    tmp_path, continuous_source
):
    offset = 500_000.0
    signal = np.arange(1024, dtype=np.float64) * 0.01
    samples = (offset + signal).astype(np.float32)
    root = _write(
        tmp_path,
        [continuous_source(samples, rate_hz=_RATE_HZ, offset_uv=offset)],
    )
    grp = root["0"]
    # Raw is the signal as recorded.
    assert np.allclose(grp["raw"][:], samples)
    # The statistics are what is left once the bias is removed.
    assert np.allclose(grp["1"]["mean"][:], _fold_mean(signal, 1), atol=1e-2)
    assert grp["1"]["env"][0, 0] == pytest.approx(signal[0], abs=1e-2)


def test_removing_the_offset_recovers_a_signal_float32_would_bury(
    tmp_path, continuous_source
):
    """Why the offset exists: an 80 uV signal on a 0.5 V bias.

    Stored with the bias in place, a float32 mean of 16384 samples cannot
    resolve the signal. Stored offset-relative, it survives.
    """
    offset = 500_000.0
    rng = np.random.default_rng(3)
    signal = rng.standard_normal(4096) * 80.0
    samples = (offset + signal).astype(np.float32)

    root = _write(
        tmp_path,
        [
            continuous_source(
                samples, id="removed", rate_hz=_RATE_HZ, offset_uv=offset
            ),
            continuous_source(samples, id="kept", rate_hz=_RATE_HZ),
        ],
    )
    expected = _fold_mean(signal, 2)
    removed = root["0"]["2"]["mean"][:].astype(np.float64)
    kept = root["1"]["2"]["mean"][:].astype(np.float64) - offset

    assert np.max(np.abs(removed - expected)) < np.max(np.abs(kept - expected))


def test_bundle_writes_mean_for_every_channel(
    tmp_path, continuous_source, unit_source
):
    samples = np.arange(1024, dtype=np.float32)
    final = tmp_path / "bundle"
    write_bundle(
        [continuous_source(samples, id="c0", rate_hz=_RATE_HZ)],
        [unit_source(np.arange(5, dtype=np.int64), id="u0")],
        staging_dir=tmp_path / "staging",
        final_dir=final,
        opts=_OPTS,
    )
    grp = open_group(final)["0"]
    for plan in _levels(samples.shape[0]):
        # A clean channel has no valid: the spec writes it only where there are gaps.
        assert sorted(grp[str(plan.level)].array_keys()) == ["env", "mean"]
