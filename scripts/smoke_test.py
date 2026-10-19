"""Check an installed timeseries-zarr wheel by writing and reading back a bundle."""

import importlib.metadata
import tempfile
import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt

import timeseries_zarr as tsz

SAMPLE_COUNT = 60_000
PROVENANCE: dict[str, object] = {"software": "smoke-test 1.0"}
WINDOW_START_US = 1_000_000
WINDOW_END_US = 3_000_000
WINDOW_START_SAMPLE = 1000
WINDOW_END_SAMPLE = 3000

warnings.filterwarnings(
    "ignore",
    message="Consolidated metadata is currently not part",
    category=UserWarning,
)


class SineSource:
    """Single continuous channel holding a 10 Hz sine sampled at 1000 Hz."""

    id = "sine-0"
    name = "sine"
    unit = "uV"

    def __init__(self) -> None:
        """Generate the samples."""
        t = np.arange(SAMPLE_COUNT) / 1000.0
        self._samples = (100.0 * np.sin(2 * np.pi * 10 * t)).astype(np.float32)

    def provenance(self) -> dict[str, object]:
        """Return the producing software."""
        return PROVENANCE

    def rate_hz(self) -> float:
        """Return the sample rate."""
        return 1000.0

    def start_us(self) -> int:
        """Return the start time."""
        return 0

    def num_samples(self) -> int:
        """Return the sample count."""
        return SAMPLE_COUNT

    def read_samples(self, start: int, stop: int) -> npt.NDArray[np.float32]:
        """Return the samples from start to stop."""
        return self._samples[start:stop]


def main() -> None:
    """Fail unless the installed package reports a consistent version and round-trips data."""
    package_dir = Path(tsz.__file__).resolve().parent
    assert "site-packages" in package_dir.parts, (
        f"not the installed wheel: {package_dir}"
    )

    assert tsz.__version__ != "0+unknown", "version file missing from the wheel"
    metadata_version = importlib.metadata.version("timeseries-zarr")
    assert tsz.__version__ == metadata_version, (
        tsz.__version__,
        metadata_version,
    )

    missing = [name for name in tsz.__all__ if not hasattr(tsz, name)]
    assert not missing, f"unresolved public names: {missing}"

    with tempfile.TemporaryDirectory() as directory:
        output_path = Path(directory) / "sine.zarr"
        tsz.write_bundle(
            [SineSource()], [], output_path=output_path, opts=tsz.WriteOpts()
        )
        with tsz.open_bundle(output_path) as bundle:
            channel = bundle.continuous.get(0)
            assert channel is not None, "channel 0 missing"
            assert channel.id == SineSource.id, (
                f"channel id {channel.id!r}, expected {SineSource.id!r}"
            )
            assert channel.provenance == PROVENANCE, (
                f"channel provenance {channel.provenance!r}, "
                f"expected {PROVENANCE!r}"
            )
            level = channel.levels.get(1)
            assert level is not None, "level 1 missing"
            samples = channel.read(WINDOW_START_US, WINDOW_END_US)
            bins = channel.read(level=1)
            whole = channel.read()
            numpy_bins = channel.read(
                np.int64(WINDOW_START_US),
                np.int64(WINDOW_END_US),
                level=np.int64(1),
            )
    expected = SineSource().read_samples(0, SAMPLE_COUNT)
    assert whole.first == 0, (
        f"whole read starts at sample {whole.first}, expected 0"
    )
    assert np.array_equal(whole.values, expected), "raw samples differ"
    assert samples.first == WINDOW_START_SAMPLE, (
        f"window starts at sample {samples.first}, "
        f"expected {WINDOW_START_SAMPLE}"
    )
    window = expected[WINDOW_START_SAMPLE:WINDOW_END_SAMPLE]
    assert np.array_equal(samples.values, window), "raw window differs"
    assert isinstance(bins, tsz.Bins), "level 1 read returned samples"
    assert len(bins) == level.bins, (
        f"level 1 read {len(bins)} bins, expected {level.bins}"
    )
    blocks = expected.reshape(-1, level.samples_per_bin)
    assert np.array_equal(bins.min, blocks.min(axis=1)), "level 1 min differs"
    assert np.array_equal(bins.max, blocks.max(axis=1)), "level 1 max differs"
    first_bin = WINDOW_START_SAMPLE // level.samples_per_bin
    stop_bin = WINDOW_END_SAMPLE // level.samples_per_bin
    assert isinstance(numpy_bins, tsz.Bins), "numpy level read returned samples"
    assert numpy_bins.first == first_bin, (
        f"numpy window starts at bin {numpy_bins.first}, expected {first_bin}"
    )
    assert np.array_equal(numpy_bins.min, bins.min[first_bin:stop_bin]), (
        "numpy window differs"
    )
    print(f"timeseries-zarr {tsz.__version__} OK")


if __name__ == "__main__":
    main()
