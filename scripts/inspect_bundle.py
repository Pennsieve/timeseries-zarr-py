"""Print what a bundle holds, and check its gaps against the source NWB.

    python scripts/inspect_bundle.py <bundle> [source.nwb]

With a source NWB it recomputes where the recording's breaks are and asserts
the bundle is NaN across them and finite inside every run.
"""

import sys
from pathlib import Path

import numpy as np
import zarr

# Run from anywhere: the package lives one level up from this script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROBE = 2000
"""Samples read at each spot check."""


def summarize(root):
    """Print one line per channel."""
    keys = sorted((k for k in root.keys() if k.isdigit()), key=int)
    print(f"{len(keys)} channels\n")
    for key in keys:
        attrs = dict(root[key].attrs)
        kind = attrs.get("kind")
        extra = ""
        if kind == "continuous":
            raw = root[key]["0"]
            levels = sorted(k for k in root[key].keys() if k.isdigit())
            extra = (
                f"{attrs.get('rate_hz')} Hz  {attrs.get('unit')}  "
                f"{raw.shape[0]:,} samples  levels {levels[0]}..{levels[-1]}"
            )
        print(f"  {key + '/':6} {kind:11} {attrs.get('name')!r:24} {extra}")
    return keys


def check_gaps(root, keys, nwb_path):
    """Assert the bundle is NaN across every break and finite inside runs."""
    import h5py

    from timeseries_zarr.grid import build_segments, derive_rate_hz

    with h5py.File(nwb_path, "r") as handle:
        timestamps = handle["acquisition/ElectricalSeries/timestamps"]
        rate = derive_rate_hz(timestamps)
        segments = build_segments(timestamps, rate)

    total = grid = 0
    for segment in segments:
        total += segment.length
    grid = segments[-1].grid_stop
    print(
        f"\nsource: {len(segments)} runs, {total:,} samples over a "
        f"{grid:,} grid ({(grid - total) / rate / 3600:.2f} h of gap)\n"
    )

    raw = root[keys[0]]["0"]
    if raw.shape[0] != grid:
        print(f"  MISMATCH: bundle raw is {raw.shape[0]:,}, expected {grid:,}")
        return False

    ok = True
    for index, segment in enumerate(segments):
        middle = (segment.grid_start + segment.grid_stop) // 2
        finite = int(np.count_nonzero(np.isfinite(raw[middle : middle + PROBE])))
        good = finite > 0
        ok &= good
        print(f"  run {index}: {finite:>5}/{PROBE} finite  {'ok' if good else 'FAIL'}")

        if index + 1 < len(segments):
            start, stop = segment.grid_stop, segments[index + 1].grid_start
            if stop - start < PROBE * 2:
                continue
            middle = (start + stop) // 2
            finite = int(
                np.count_nonzero(np.isfinite(raw[middle : middle + PROBE]))
            )
            good = finite == 0
            ok &= good
            print(
                f"  gap {index}: {finite:>5}/{PROBE} finite  "
                f"{'ok (all NaN)' if good else 'FAIL -- expected all NaN'}"
            )
    return bool(ok)


def main():
    """Summarize the bundle, then check gaps when a source NWB is given."""
    bundle = Path(sys.argv[1])
    root = zarr.open_group(str(bundle), mode="r")
    print(f"{bundle.name}\n")
    keys = summarize(root)

    if len(sys.argv) > 2:
        if not check_gaps(root, keys, Path(sys.argv[2])):
            print("\nFAILED")
            sys.exit(1)
        print("\nOK")


if __name__ == "__main__":
    main()
