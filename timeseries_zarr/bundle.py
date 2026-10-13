"""Top-level bundle orchestration."""

import shutil
from collections.abc import Sequence
from pathlib import Path

from timeseries_zarr.protocols import ContinuousChannelSource, UnitChannelSource
from timeseries_zarr.types import WriteOpts
from timeseries_zarr.write_continuous import write_continuous_channel
from timeseries_zarr.write_unit import write_unit_channel
from timeseries_zarr.zarr_io import ZarrGroup, consolidate, open_group


def assign_indices(
    continuous: Sequence[ContinuousChannelSource],
    units: Sequence[UnitChannelSource],
) -> list[tuple[int, ContinuousChannelSource | UnitChannelSource]]:
    """Assign each channel a digit index, continuous first then unit.

    Indices are contiguous from 0 and become the bundle's digit-named
    channel-group directories. Sources keep their input order within each kind.
    """
    return list(enumerate([*continuous, *units]))


def atomic_publish(staging_dir: Path, final_dir: Path) -> None:
    """Move a fully-staged bundle to its final path in one atomic rename.

    Replaces final_dir if it already exists. A reader never sees a partial
    write: it sees the old bundle, the complete new one, or nothing for the
    brief window while final_dir is replaced. Both paths must be on the same
    filesystem; a cross-device move raises OSError.
    """
    if not final_dir.exists():
        staging_dir.replace(final_dir)
        return

    # A single os.replace cannot overwrite a non-empty directory, so the old
    # bundle is moved aside first and survives until the swap succeeds.
    backup = final_dir.with_name(final_dir.name + ".old")
    final_dir.replace(backup)
    try:
        staging_dir.replace(final_dir)
    except OSError:
        backup.replace(final_dir)
        raise
    shutil.rmtree(backup)


def write_all_channels(
    root: ZarrGroup,
    indexed: Sequence[tuple[int, ContinuousChannelSource | UnitChannelSource]],
    opts: WriteOpts,
) -> None:
    """Write every indexed channel under root, dispatching by source type.

    Each (index, source) pair becomes its own channel group beneath root,
    governed by opts.
    """
    origin_us = min((source.start_us() for _, source in indexed), default=0)
    for index, source in indexed:
        if isinstance(source, UnitChannelSource):
            write_unit_channel(root, index, source, opts=opts)
        else:
            write_continuous_channel(
                root, index, source, origin_us=origin_us, opts=opts
            )


def write_bundle(
    continuous: Sequence[ContinuousChannelSource],
    units: Sequence[UnitChannelSource],
    *,
    staging_dir: Path,
    final_dir: Path,
    opts: WriteOpts,
) -> None:
    """Build the whole viewer bundle and publish it atomically.

    Every channel is staged into a fresh root group at staging_dir before the
    rename onto final_dir. Sizing and compression follow opts.
    """
    root = open_group(staging_dir)
    write_all_channels(root, assign_indices(continuous, units), opts)
    consolidate(root)
    atomic_publish(staging_dir, final_dir)
