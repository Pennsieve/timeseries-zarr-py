"""Top-level bundle orchestration."""

import shutil
from collections.abc import Sequence
from pathlib import Path

from timeseries_zarr.attrs import meta_group_attrs
from timeseries_zarr.protocols import (
    AnnotationChannelSource,
    ContinuousChannelSource,
    UnitChannelSource,
)
from timeseries_zarr.types import RecordingMeta, WriteOpts
from timeseries_zarr.write_annotation import write_annotation_channel
from timeseries_zarr.write_continuous import write_continuous_channel
from timeseries_zarr.write_unit import write_unit_channel
from timeseries_zarr.zarr_io import (
    ZarrGroup,
    consolidate,
    open_group,
    write_meta_group,
)

type AnyChannelSource = (
    ContinuousChannelSource | UnitChannelSource | AnnotationChannelSource
)
"""Any channel a bundle can hold, whatever writer handles it."""


def assign_indices(
    continuous: Sequence[ContinuousChannelSource],
    units: Sequence[UnitChannelSource],
    annotations: Sequence[AnnotationChannelSource] = (),
) -> list[tuple[int, AnyChannelSource]]:
    """Assign each channel a digit index: continuous, then unit, then annotation.

    Indices are contiguous from 0 and become the bundle's digit-named
    channel-group directories. Sources keep their input order within each kind.

    The order is the one thing channel_refs depends on. An annotation naming
    a continuous channel names its index, so anything that renumbers channels
    between this call and the write silently repoints every reference.
    """
    return list(enumerate([*continuous, *units, *annotations]))


def bundle_onset_us(
    sources: Sequence[AnyChannelSource],
) -> int:
    """Return the bundle's onset: the earliest wall-clock start of any channel.

    Every channel then records its distance from this one instant, and the
    instant itself is written only into meta/. Taking the earliest keeps
    every offset at or above zero. A bundle with no channels has no onset
    and reports 0.
    """
    return min((source.start_us() for source in sources), default=0)


def channel_index_by_name(
    continuous: Sequence[ContinuousChannelSource],
    units: Sequence[UnitChannelSource],
) -> dict[str, int]:
    """Return the display name of each continuous channel to its index.

    What an annotation's channel references resolve against. It shares
    assign_indices rather than recomputing the order, because the two
    disagreeing would repoint every reference in a bundle silently.
    """
    return {
        source.name: index
        for index, source in assign_indices(continuous, units)
        if isinstance(source, ContinuousChannelSource)
        and not isinstance(source, UnitChannelSource)
    }


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
    indexed: Sequence[tuple[int, AnyChannelSource]],
    onset_us: int,
    opts: WriteOpts,
) -> None:
    """Write every indexed channel under root, dispatching by source type.

    Each (index, source) pair becomes its own channel group beneath root,
    governed by opts and placed relative to onset_us.
    """
    for index, source in indexed:
        if isinstance(source, UnitChannelSource):
            write_unit_channel(
                root, index, source, onset_us=onset_us, opts=opts
            )
        elif isinstance(source, AnnotationChannelSource):
            write_annotation_channel(
                root, index, source, onset_us=onset_us, opts=opts
            )
        else:
            write_continuous_channel(
                root, index, source, onset_us=onset_us, opts=opts
            )


def write_bundle(
    continuous: Sequence[ContinuousChannelSource],
    units: Sequence[UnitChannelSource],
    *,
    annotations: Sequence[AnnotationChannelSource] = (),
    staging_dir: Path,
    final_dir: Path,
    opts: WriteOpts,
    meta: RecordingMeta | None = None,
) -> None:
    """Build the whole viewer bundle and publish it atomically.

    Every channel is staged into a fresh root group at staging_dir before the
    rename onto final_dir. Sizing and compression follow opts.

    The bundle's timeline is relative to the earliest channel start, and that
    one wall-clock instant is written into meta/ as session.start_us. meta is
    optional: without it the bundle is a pure relative timeline, which stays
    fully reviewable and carries no identity at all.

    1. Stage every channel, placed relative to the onset.
    2. Consolidate, which inlines every descendant into the root object.
    3. Write meta/ afterwards, so consolidation cannot copy identity into
       that root object and deleting the directory stays sufficient.
    4. Publish.
    """
    indexed = assign_indices(continuous, units, annotations)
    onset_us = bundle_onset_us([source for _, source in indexed])

    root = open_group(staging_dir)
    write_all_channels(root, indexed, onset_us, opts)
    consolidate(root)
    if meta is not None:
        write_meta_group(
            staging_dir,
            meta_group_attrs(
                meta.subject,
                {**meta.session, "start_us": onset_us},
                meta.source,
            ),
        )
    atomic_publish(staging_dir, final_dir)
