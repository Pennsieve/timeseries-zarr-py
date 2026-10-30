"""Check a bundle against docs/spec.md from the command line.

Run python -m timeseries_zarr.validate PATH [--full]. The command prints one
finding per line and exits with status 1 when it finds any. The checks live
in timeseries_zarr.validator.
"""

import argparse
from collections.abc import Sequence

from timeseries_zarr.types import Finding
from timeseries_zarr.validator import validate_bundle

__all__ = ["Finding", "main", "validate_bundle"]


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the bundle that argv names and print the findings.

    Args:
        argv: The command line arguments. None reads sys.argv.

    Returns:
        0 when the bundle has no findings, otherwise 1.
    """
    parser = argparse.ArgumentParser(
        prog="python -m timeseries_zarr.validate",
        description="Check a TS-Zarr bundle against the specification.",
    )
    parser.add_argument("path", help="local path or URL of the bundle")
    parser.add_argument(
        "--full",
        action="store_true",
        help="also recompute the statistics from the samples and events",
    )
    arguments = parser.parse_args(argv)
    findings = validate_bundle(arguments.path, full=arguments.full)
    for finding in findings:
        print(
            f"{finding.path or '.'}: section {finding.section}: "
            f"{finding.message}"
        )
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
