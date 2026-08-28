"""``tszarr``: inspect, validate, and de-identify TS-Zarr bundles.

Subcommands::

    tszarr info <bundle>                channel inventory
    tszarr validate <bundle>            spec conformance (exit 1 on error)
    tszarr deid <bundle>                delete meta/, list body surfaces
    tszarr dateshift <bundle> --days N  shift the one wall-clock value
"""

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime

from timeseries_zarr.deid import dateshift, deidentify
from timeseries_zarr.reading.bundle import (
    ContinuousChannel,
    EventChannel,
    open_bundle,
)
from timeseries_zarr.validate import errors, validate_bundle


def _cmd_info(path: str) -> int:
    b = open_bundle(path)
    meta = b.meta
    start = b.start_us
    if start is not None:
        stamp = datetime.fromtimestamp(start / 1e6, tz=UTC).isoformat()
        print(f"onset: {stamp} (meta/session.start_us)")
    elif meta is None:
        print("onset: relative only (no meta/ -- de-identified)")
    for key, ch in sorted(b.channels.items()):
        if isinstance(ch, ContinuousChannel):
            lv = ",".join(str(level.k) for level in ch.levels)
            raw = "raw" if ch.has_raw else "viewing-only"
            print(
                f"  {key}: continuous {ch.name!r} "
                f"{ch.rate_hz:g} Hz, {ch.duration_s:.1f} s, "
                f"{raw}, levels [{lv}]"
            )
        elif isinstance(ch, EventChannel):
            cols = [
                c
                for c in ("durations", "labels", "values", "bodies",
                          "waveforms")
                if c in ch.group.array_keys()
            ]
            lv = ",".join(str(level.k) for level in ch.levels)
            print(
                f"  {key}: event {ch.name!r} "
                f"{ch.n_events} events, columns {cols}, "
                f"count levels [{lv}]"
            )
    return 0


def _cmd_validate(path: str) -> int:
    findings = validate_bundle(path)
    for f in findings:
        print(f)
    errs = errors(findings)
    n_warn = sum(1 for f in findings if f.severity == "warning")
    print(f"{len(errs)} error(s), {n_warn} warning(s)")
    return 1 if errs else 0


def _cmd_deid(path: str) -> int:
    surfaces = deidentify(path)
    print(f"removed meta/ from {path}")
    if surfaces:
        print("remaining human-readable surfaces to review:")
        for s in surfaces:
            print(
                f"  channel {s.channel} ({s.name!r}): "
                f"{s.n_bodies} bodies [{s.media_type}] at {s.path}"
            )
    else:
        print("no annotation bodies stored; nothing left to review")
    return 0


def _cmd_dateshift(path: str, days: float) -> int:
    new = dateshift(path, int(days * 86_400e6))
    stamp = datetime.fromtimestamp(new / 1e6, tz=UTC).isoformat()
    print(f"meta/session.start_us -> {new} ({stamp})")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return an exit code."""
    parser = argparse.ArgumentParser(
        prog="tszarr", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    for cmd in ("info", "validate", "deid"):
        p = sub.add_parser(cmd)
        p.add_argument("bundle")
    p = sub.add_parser("dateshift")
    p.add_argument("bundle")
    p.add_argument("--days", type=float, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "info":
        return _cmd_info(args.bundle)
    if args.cmd == "validate":
        return _cmd_validate(args.bundle)
    if args.cmd == "deid":
        return _cmd_deid(args.bundle)
    return _cmd_dateshift(args.bundle, args.days)


if __name__ == "__main__":
    sys.exit(main())
