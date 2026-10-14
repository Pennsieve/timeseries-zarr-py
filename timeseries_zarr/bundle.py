"""Top-level bundle orchestration."""

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from timeseries_zarr.attrs import root_attrs
from timeseries_zarr.constants import ZARR_METADATA_FILE
from timeseries_zarr.protocols import (
    ContinuousChannelSource,
    EventChannelSource,
)
from timeseries_zarr.recording import (
    recording_document,
    session_start_us,
    write_recording_json,
)
from timeseries_zarr.types import RecordingMeta, WriteOpts
from timeseries_zarr.units import resolve_unit
from timeseries_zarr.write_continuous import write_continuous_channel
from timeseries_zarr.write_event import write_event_channel
from timeseries_zarr.zarr_io import ZarrGroup, consolidate, open_group

type AnySource = ContinuousChannelSource | EventChannelSource
"""Any channel source the bundle writes, whichever protocol it follows."""


def assign_indices(
    continuous: Sequence[ContinuousChannelSource],
    events: Sequence[EventChannelSource],
) -> list[tuple[int, AnySource]]:
    """Assign each channel a digit index, continuous first then event.

    Indices are contiguous from 0 and become the bundle's digit-named
    channel-group directories. Sources keep their input order within each kind.
    """
    return list(enumerate([*continuous, *events]))


def origin_us(
    sources: Sequence[AnySource],
    session_start_us: int | None = None,
) -> int:
    """Return the bundle origin: the earliest wall-clock instant it holds.

    Every channel and event is placed in microseconds from this instant, so it
    is the minimum of the session start, when given, and the start of every
    source. No offset or event time is then negative. The origin is the one
    wall-clock value the writer maps to the time zero of the bundle. A bundle
    with neither a session start nor a channel has no instant to count from and
    reports 0.
    """
    starts = [source.start_us() for source in sources]
    if session_start_us is not None:
        starts.append(session_start_us)
    return min(starts, default=0)


def channel_keys(indexed: Sequence[tuple[int, AnySource]]) -> dict[str, int]:
    """Return each channel's id mapped to its channel key.

    The key is the index assign_indices gave the channel, which names its group
    in the bundle. An event channel names the channels it applies to by id, so
    this is the table a channels column resolves through. Raises ValueError if
    two channels share an id.
    """
    keys: dict[str, int] = {}
    for index, source in indexed:
        if source.id in keys:
            raise ValueError(
                f"channels {keys[source.id]} and {index} share the id "
                f"{source.id!r}"
            )
        keys[source.id] = index
    return keys


def max_key(indexed: Sequence[tuple[int, AnySource]]) -> int:
    """Return the largest channel key, which sizes the channels bitmask.

    A bundle with no channels reports 0.
    """
    return max((index for index, _ in indexed), default=0)


def clear_staging(staging_dir: Path) -> None:
    """Remove a bundle an interrupted run left at staging_dir.

    Does nothing when staging_dir is absent or empty. A staged bundle has a
    root zarr.json. Raises ValueError if staging_dir holds anything else, and
    removes nothing, so a mistyped path loses no files.
    """
    if not staging_dir.is_dir() or not any(staging_dir.iterdir()):
        return
    if not (staging_dir / ZARR_METADATA_FILE).is_file():
        raise ValueError(
            f"staging_dir {staging_dir} holds files but no staged bundle; "
            "use an empty or absent directory"
        )
    shutil.rmtree(staging_dir)


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
    indexed: Sequence[tuple[int, AnySource]],
    origin_us: int,
    channel_keys: Mapping[str, int],
    opts: WriteOpts,
) -> None:
    """Write every indexed channel under root, dispatching by source type.

    Each (index, source) pair becomes its own channel group beneath root,
    governed by opts and placed relative to origin_us. channel_keys maps
    channel ids to keys, which the event channels use to resolve the channels
    they apply to.
    """
    largest_key = max_key(indexed)
    for index, source in indexed:
        if isinstance(source, EventChannelSource):
            write_event_channel(
                root,
                index,
                source,
                origin_us=origin_us,
                channel_keys=channel_keys,
                max_key=largest_key,
                opts=opts,
            )
        else:
            write_continuous_channel(
                root, index, source, origin_us=origin_us, opts=opts
            )


def write_bundle(
    continuous: Sequence[ContinuousChannelSource],
    events: Sequence[EventChannelSource],
    *,
    staging_dir: Path,
    final_dir: Path,
    opts: WriteOpts,
    recording: RecordingMeta | None = None,
) -> None:
    """Build the whole viewer bundle and publish it atomically.

    The root group carries the format version. Every channel is staged into a
    fresh root group at staging_dir, the metadata of every node is consolidated
    into the root, and then recording.json is written when recording is given,
    so identity and the wall-clock start never enter the root object. The
    staged bundle is renamed onto final_dir last. A bundle an interrupted run
    left at staging_dir is removed first. Every time counts from the origin,
    the earliest of the session start in recording and every channel start,
    and recording.json holds that instant as its start. Sizing and compression
    follow opts. Raises ValueError if two channels share an id,
    recording.start has no UTC offset, a voltage unit is not "uV" under
    voltage_policy require_uv, or staging_dir holds files that are not
    a staged bundle, before anything is written.
    """
    indexed = assign_indices(continuous, events)
    session_start = None if recording is None else session_start_us(recording)
    origin = origin_us([source for _, source in indexed], session_start)
    keys = channel_keys(indexed)
    for index, source in indexed:
        if not isinstance(source, EventChannelSource):
            try:
                resolve_unit(source.unit, opts.voltage_policy)
            except ValueError as error:
                raise ValueError(
                    f"channel {index} ({source.id!r}): {error}"
                ) from error
    clear_staging(staging_dir)
    root = open_group(staging_dir, root_attrs())
    write_all_channels(root, indexed, origin, keys, opts)
    consolidate(root)
    if recording is not None:
        write_recording_json(staging_dir, recording_document(recording, origin))
    atomic_publish(staging_dir, final_dir)
