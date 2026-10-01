"""Run configuration assembled from the environment and command line."""

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from timeseries_zarr.constants import MAX_LEVELS
from timeseries_zarr.properties import DEFAULT_PROPERTIES_FILE
from timeseries_zarr.types import WriteOpts


@dataclass(frozen=True, slots=True)
class Config:
    """Everything one bundle run needs, resolved from env and argv.

    staging_dir is the scratch path the bundle is built in before its atomic
    rename onto final_dir. properties_path is the output properties file, a
    sibling of final_dir.
    """

    nwb_path: Path
    annotation_paths: tuple[Path, ...]
    staging_dir: Path
    final_dir: Path
    properties_path: Path
    opts: WriteOpts


def load_config(env: Mapping[str, str], argv: Sequence[str]) -> Config:
    """Resolve a Config from environment variables and command-line arguments.

    argv holds the arguments after the program name: two positionals are the
    input NWB path and the final output directory. With no positionals the
    paths come from INPUT_DIR and OUTPUT_DIR, INPUT_DIR being scanned for the
    single *.nwb file it must hold. The writer settings and the staging
    directory come from the ZARR_WRITER_ prefixed variables; unset settings
    fall back to the WriteOpts defaults and an unset staging directory derives
    from final_dir. ASSET_PROPERTIES_FILE names the output properties file
    beside the bundle. Raises ValueError on a bad invocation: one positional,
    INPUT_DIR or OUTPUT_DIR unset, an INPUT_DIR without exactly one *.nwb
    file, a non-integer setting, or ZARR_WRITER_MAX_LEVELS outside 1 through
    MAX_LEVELS.
    """
    nwb_path, final_dir = _resolve_paths(env, argv)
    annotation_paths = _resolve_annotation_paths(env, argv, nwb_path)

    defaults = WriteOpts()
    max_levels = _int_env(env, "ZARR_WRITER_MAX_LEVELS", defaults.max_levels)
    if not 1 <= max_levels <= MAX_LEVELS:
        raise ValueError(
            f"ZARR_WRITER_MAX_LEVELS must be 1 to {MAX_LEVELS}, "
            f"got {max_levels}"
        )
    opts = WriteOpts(
        zstd_level=_int_env(env, "ZARR_WRITER_ZSTD_LEVEL", defaults.zstd_level),
        max_levels=max_levels,
        min_bins=_int_env(env, "ZARR_WRITER_MIN_BINS", defaults.min_bins),
        inner_len=_int_env(env, "ZARR_WRITER_INNER_LEN", defaults.inner_len),
        target_shard_bytes=_int_env(
            env, "ZARR_WRITER_TARGET_SHARD_BYTES", defaults.target_shard_bytes
        ),
    )

    staging_raw = env.get("ZARR_WRITER_STAGING_DIR")
    staging_dir = (
        Path(staging_raw)
        if staging_raw is not None
        else final_dir.with_name(final_dir.name + ".staging")
    )

    properties_name = env.get("ASSET_PROPERTIES_FILE", DEFAULT_PROPERTIES_FILE)

    return Config(
        nwb_path=nwb_path,
        annotation_paths=annotation_paths,
        staging_dir=staging_dir,
        final_dir=final_dir,
        properties_path=final_dir.parent / properties_name,
        opts=opts,
    )


def _resolve_paths(
    env: Mapping[str, str], argv: Sequence[str]
) -> tuple[Path, Path]:
    """Return the (input NWB, final output dir) paths from argv or env.

    Two positionals take precedence over the environment. Raises ValueError on
    any other argv shape.
    """
    positional_count = 2
    if len(argv) >= positional_count:
        return Path(argv[0]), Path(argv[1])
    if len(argv) == 1:
        raise ValueError(
            "load_config needs an input NWB path and an output directory"
        )

    input_dir = env.get("INPUT_DIR")
    output_dir = env.get("OUTPUT_DIR")
    if input_dir is None or output_dir is None:
        raise ValueError(
            "load_config needs two positionals or INPUT_DIR and OUTPUT_DIR"
        )
    # Atomic publish renames the final path, and a rename cannot target a mount
    # point such as the bare OUTPUT_DIR volume. The bundle is therefore a named
    # directory inside it, taking the input stem: session.nwb -> session.zarr.
    nwb_path = _sole_nwb(Path(input_dir))
    return nwb_path, Path(output_dir) / f"{nwb_path.stem}.zarr"


def _sole_nwb(input_dir: Path) -> Path:
    """Return the single *.nwb file in input_dir.

    Raises ValueError unless exactly one *.nwb file is present.
    """
    nwbs = sorted(
        Path(entry.path)
        for entry in os.scandir(input_dir)
        if entry.is_file() and entry.name.lower().endswith(".nwb")
    )
    if len(nwbs) != 1:
        raise ValueError(
            f"expected exactly one .nwb file in {input_dir}, found {len(nwbs)}"
        )
    return nwbs[0]


def _int_env(env: Mapping[str, str], key: str, default: int) -> int:
    """Return the integer env value for key, or default if it is unset.

    Raises ValueError if the value is present but not an integer.
    """
    raw = env.get(key)
    return default if raw is None else int(raw)


def _resolve_annotation_paths(
    env: Mapping[str, str], argv: Sequence[str], nwb_path: Path
) -> tuple[Path, ...]:
    """Return the annotation documents to write alongside the recording.

    Any positional after the first two is one file. With none given, the
    NWB's own directory is scanned for *.annotations.ndjson, which is the
    convention the workflow runs on: the extractor branch drops its output
    beside the recording at the merge and the writer picks it up without
    being told.

    One file is one event channel, and the order here is the order they are
    indexed in.
    """
    explicit = [Path(argument) for argument in argv[2:]]
    if explicit:
        return tuple(explicit)
    directory = Path(env.get("INPUT_DIR", "")) or nwb_path.parent
    return tuple(sorted(directory.glob("*.annotations.ndjson")))
