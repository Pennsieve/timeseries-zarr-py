"""Check an installed timeseries-zarr wheel by writing and reading back a bundle."""

import importlib.metadata
import tempfile
import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt
import zarr

import timeseries_zarr as tsz

SAMPLE_COUNT = 60_000
PROVENANCE: dict[str, object] = {"software": "smoke-test 1.0"}

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
        root = zarr.open_group(str(output_path), mode="r")
        channel = root.get("0")
        assert isinstance(channel, zarr.Group), "channel group missing"
        assert dict(channel.attrs["tsz"]).get("id") == SineSource.id  # type: ignore[arg-type]
        assert dict(channel.attrs["tsz"]).get("provenance") == PROVENANCE  # type: ignore[arg-type]
        assert "1" in channel.group_keys(), "level 1 missing"
        raw_array = channel.get("raw")
        assert isinstance(raw_array, zarr.Array), "raw array missing"
        raw = np.asarray(raw_array[:])
    expected = SineSource().read_samples(0, SAMPLE_COUNT)
    assert np.array_equal(raw, expected), "raw samples differ"
    print(f"timeseries-zarr {tsz.__version__} OK")


if __name__ == "__main__":
    main()
