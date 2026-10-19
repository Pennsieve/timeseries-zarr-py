"""Top-level bundle orchestration."""

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from timeseries_zarr.attrs import check_provenance, root_attrs
from timeseries_zarr.constants import TSZ_KEY, ZARR_METADATA_FILE
from timeseries_zarr.protocols import (
    ContinuousChannelSource,
    EventChannelSource,
    source_provenance,
)
from timeseries_zarr.recording import recording_document, write_recording_json
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


def recording_start_us(sources: Sequence[AnySource]) -> int | None:
    """Return the recording start in wall-clock microseconds since the epoch.

    The recording start is the earliest sample or event the sources hold: the
    minimum of every continuous source's start and of the first event of
    every event source that has one. Every offset and event time counts from
    it, so the smallest is 0 and none is negative. Sources with no sample and
    no event give None.
    """
    starts: list[int] = []
    for source in sources:
        if isinstance(source, EventChannelSource):
            if source.num_events():
                starts.append(int(source.read_events(0, 1)[0]))
        else:
            starts.append(source.start_us())
    return min(starts, default=None)


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
    recording_start_us: int,
    channel_keys: Mapping[str, int],
    opts: WriteOpts,
) -> None:
    """Write every indexed channel under root, dispatching by source type.

    Each (index, source) pair becomes its own channel group beneath root,
    governed by opts and placed relative to recording_start_us. channel_keys maps
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
                recording_start_us=recording_start_us,
                channel_keys=channel_keys,
                max_key=largest_key,
                opts=opts,
            )
        else:
            write_continuous_channel(
                root,
                index,
                source,
                recording_start_us=recording_start_us,
                opts=opts,
            )


def write_bundle(
    continuous: Sequence[ContinuousChannelSource],
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
    recording is given, so identity and the wall-clock times never enter the
    root object. A bundle an earlier run left at output_path is replaced.
    Every time counts from the recording start, the earliest sample or event
    the channels hold. recording.json holds it as start_time and the session
    start of recording as session.start_time. Sizing and compression follow
    opts. Raises ValueError if two channels share an id, recording has a
    session start without a UTC offset or a session object with a start_time
    key, a voltage unit is not "uV" under voltage_policy require_uv, a
    channel's provenance is not a JSON object of finite values, or
    output_path is not absent, empty, or a bundle, before anything is
    written. A failed write leaves a partial bundle with no consolidated
    metadata.
    """
    indexed = assign_indices(continuous, events)
    start = recording_start_us([source for _, source in indexed])
    document = (
        None if recording is None else recording_document(recording, start)
    )
    keys = channel_keys(indexed)
    for index, source in indexed:
        try:
            if not isinstance(source, EventChannelSource):
                resolve_unit(source.unit, opts.voltage_policy)
            check_provenance(source_provenance(source))
        except ValueError as error:
            raise ValueError(
                f"channel {index} ({source.id!r}): {error}"
            ) from error
    clear_output_path(output_path)
    root = open_group(output_path, root_attrs())
    write_all_channels(root, indexed, 0 if start is None else start, keys, opts)
    consolidate(root)
    if document is not None:
        write_recording_json(output_path, document)
