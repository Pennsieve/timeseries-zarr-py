"""Top-level bundle orchestration."""

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from timeseries_zarr.attrs import check_provenance, root_attrs
from timeseries_zarr.constants import TSZ_KEY, ZARR_METADATA_FILE
from timeseries_zarr.protocols import (
    ContinuousChannelMetadata,
    ContinuousChannelSource,
    EventChannelSource,
    MultiChannelSource,
    source_provenance,
)
from timeseries_zarr.recording import (
    recording_document,
    reference_time_us,
    session_start_us,
    write_recording_json,
)
from timeseries_zarr.types import RecordingMeta, WriteOpts
from timeseries_zarr.units import resolve_unit
from timeseries_zarr.write_continuous import (
    write_continuous_channel,
    write_continuous_channels,
)
from timeseries_zarr.write_event import write_event_channel
from timeseries_zarr.zarr_io import ZarrGroup, consolidate, open_group

type AnySource = (
    ContinuousChannelSource | MultiChannelSource | EventChannelSource
)
"""Any channel source the bundle writes, whichever protocol it follows."""

type AnyChannel = ContinuousChannelMetadata | EventChannelSource
"""One channel of a bundle, whichever source it comes from."""


def assign_indices(
    continuous: Sequence[ContinuousChannelSource | MultiChannelSource],
    events: Sequence[EventChannelSource],
) -> list[tuple[int, AnySource]]:
    """Assign each source its first channel key, continuous first then event.

    Keys are contiguous from 0 and become the bundle's digit-named
    channel-group directories. A multi-channel source is one entry at its first
    key, and the next source takes the key after its last channel. Sources keep
    their input order within each kind.
    """
    indexed: list[tuple[int, AnySource]] = []
    key = 0
    for source in continuous:
        indexed.append((key, source))
        key += (
            len(source.channels())
            if isinstance(source, MultiChannelSource)
            else 1
        )
    for event_source in events:
        indexed.append((key, event_source))
        key += 1
    return indexed


def indexed_channels(
    indexed: Sequence[tuple[int, AnySource]],
) -> list[tuple[int, AnyChannel]]:
    """List every channel of the indexed sources at its own key.

    A multi-channel source gives one pair for each of its channels, at
    consecutive keys from its entry's key. Any other entry stays as it is, and
    a list without a multi-channel source comes back equal.
    """
    channels: list[tuple[int, AnyChannel]] = []
    for key, source in indexed:
        if isinstance(source, MultiChannelSource):
            channels.extend(enumerate(source.channels(), start=key))
        else:
            channels.append((key, source))
    return channels


def reference_us(
    sources: Sequence[AnyChannel],
    session_start_us: int | None = None,
    declared_us: int | None = None,
) -> int | None:
    """Return the reference time in microseconds since the Unix epoch.

    Every offset and event time counts from it. declared_us, when given, is
    the reference time as it is, so a rewrite keeps a bundle's reference time.
    Otherwise it is the earliest of the session start, when given, and the
    first sample of every continuous source that holds samples, so no sample
    precedes it. A continuous source without samples has no first sample and
    does not count. Without either, it is the first event of the event
    sources that have one, and without any of those it is None.
    """
    if declared_us is not None:
        return declared_us
    starts = [
        source.start_us()
        for source in sources
        if not isinstance(source, EventChannelSource) and source.num_samples()
    ]
    if session_start_us is not None:
        starts.append(session_start_us)
    if starts:
        return min(starts)
    first_events = [
        int(source.read_events(0, 1)[0])
        for source in sources
        if isinstance(source, EventChannelSource) and source.num_events()
    ]
    return min(first_events, default=None)


def channel_keys(indexed: Sequence[tuple[int, AnySource]]) -> dict[str, int]:
    """Return each channel's id mapped to its channel key.

    The key names the channel's group in the bundle, and each channel of a
    multi-channel source has its own. An event channel names the channels it
    applies to by id, so this is the table a channels column resolves through.
    Raises ValueError if two channels share an id.
    """
    keys: dict[str, int] = {}
    for index, channel in indexed_channels(indexed):
        if channel.id in keys:
            raise ValueError(
                f"channels {keys[channel.id]} and {index} share the id "
                f"{channel.id!r}"
            )
        keys[channel.id] = index
    return keys


def max_key(indexed: Sequence[tuple[int, AnySource]]) -> int:
    """Return the largest channel key, which sizes the channels bitmask.

    Every channel of a multi-channel source counts. A bundle with no channels
    reports 0.
    """
    return max((index for index, _ in indexed_channels(indexed)), default=0)


def clear_output_path(output_path: Path) -> None:
    """Empty output_path of a bundle an earlier run left there.

    Does nothing when output_path is absent or empty. Keeps the directory
    itself and removes its root zarr.json last, so an interrupted clear still
    leaves a bundle the next run recognizes. Raises ValueError, and removes
    nothing, if output_path is not a directory or holds files but no bundle.
    """
    if not output_path.exists():
        return
    if not output_path.is_dir():
        raise ValueError(
            f"output_path {output_path} is not a directory; "
            "use an absent path, an empty directory, or an existing bundle"
        )
    entries = list(output_path.iterdir())
    if not entries:
        return
    root = output_path / ZARR_METADATA_FILE
    if not _is_bundle_root(root):
        raise ValueError(
            f"output_path {output_path} holds files but no bundle; "
            "use an absent path, an empty directory, or an existing bundle"
        )
    for entry in entries:
        if entry == root:
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    root.unlink()


def _is_bundle_root(path: Path) -> bool:
    """Return whether path is a root zarr.json that carries tsz.version."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(document, dict):
        return False
    attributes = document.get("attributes")
    if not isinstance(attributes, dict):
        return False
    tsz = attributes.get(TSZ_KEY)
    return isinstance(tsz, dict) and isinstance(tsz.get("version"), str)


def write_all_channels(
    root: ZarrGroup,
    indexed: Sequence[tuple[int, AnySource]],
    reference_us: int,
    channel_keys: Mapping[str, int],
    opts: WriteOpts,
) -> None:
    """Write every indexed source under root, dispatching by source type.

    A single channel becomes the channel group named for its index. A
    multi-channel source becomes one channel group for each of its channels, at
    consecutive keys from its index. Every group is governed by opts and placed
    relative to reference_us. channel_keys maps channel ids to keys, which the
    event channels use to resolve the channels they apply to.
    """
    largest_key = max_key(indexed)
    for index, source in indexed:
        if isinstance(source, EventChannelSource):
            write_event_channel(
                root,
                index,
                source,
                reference_us=reference_us,
                channel_keys=channel_keys,
                max_key=largest_key,
                opts=opts,
            )
        elif isinstance(source, MultiChannelSource):
            write_continuous_channels(
                root,
                index,
                source,
                reference_us=reference_us,
                opts=opts,
            )
        else:
            write_continuous_channel(
                root,
                index,
                source,
                reference_us=reference_us,
                opts=opts,
            )


def write_bundle(
    continuous: Sequence[ContinuousChannelSource | MultiChannelSource],
    events: Sequence[EventChannelSource],
    *,
    output_path: Path,
    opts: WriteOpts,
    recording: RecordingMeta | None = None,
) -> None:
    """Write the whole viewer bundle at output_path.

    The root group carries the format version. Every channel is written into
    a fresh root group at output_path, the metadata of every node is
    consolidated into the root, and then recording.json is written when
    recording is given, so identity and the datetimes never enter the
    root object. A bundle an earlier run left at output_path is replaced.
    Each channel has a key that names its group. Keys count up from 0 through
    the continuous sources in list order and then the event sources. A
    multi-channel source takes one key for each channel, in channel order.
    Every time counts from the reference time: recording.reference_time when
    it is given, else the earlier of the session start and the first sample,
    as reference_us resolves it. A channel or event before it has a negative
    time. recording.json holds the reference time as reference_time and the
    session start as session.start_time. With recording.wall_clock False the
    source times do not count from the Unix epoch, so the session start does
    not place the reference time. Sizing and compression follow opts.
    Raises ValueError if two channels share an id, a time of recording has no
    UTC offset, recording sets reference_time with wall_clock False,
    recording.session holds a start_time key, a voltage unit is not "uV"
    under voltage_policy require_uv, a
    channel's provenance is not a JSON object of finite values, or
    output_path is not absent, empty, or a bundle, before anything is
    written. Raises ValueError during the write if a window of a
    multi-channel source holds the wrong number of blocks, a block is not
    rank 1, or the blocks of a channel do not total its num_samples. A failed
    write leaves a partial bundle with no consolidated metadata.
    """
    indexed = assign_indices(continuous, events)
    channels = indexed_channels(indexed)
    sources = [channel for _, channel in channels]
    if recording is None:
        reference = reference_us(sources)
        document = None
    else:
        session_start = (
            session_start_us(recording) if recording.wall_clock else None
        )
        reference = reference_us(
            sources, session_start, reference_time_us(recording)
        )
        document = recording_document(recording, reference)
    keys = channel_keys(indexed)
    for index, channel in channels:
        try:
            if not isinstance(channel, EventChannelSource):
                resolve_unit(channel.unit, opts.voltage_policy)
            check_provenance(source_provenance(channel))
        except ValueError as error:
            raise ValueError(
                f"channel {index} ({channel.id!r}): {error}"
            ) from error
    clear_output_path(output_path)
    root = open_group(output_path, root_attrs())
    write_all_channels(
        root, indexed, 0 if reference is None else reference, keys, opts
    )
    consolidate(root)
    if document is not None:
        write_recording_json(output_path, document)
