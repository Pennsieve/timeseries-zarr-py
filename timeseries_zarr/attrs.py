"""Build the attribute dicts written into the bundle's zarr.json files.

The format spec specifies the keys and their types:
https://github.com/Pennsieve/timeseries-zarr-paper/blob/main/bundle-format.md
"""

from timeseries_zarr.types import ChannelKind


def root_attrs() -> dict[str, object]:
    """Return the custom attributes for the bundle's root group.

    Always empty. The root carries only Zarr's own consolidated_metadata.
    """
    return {}


def channel_group_attrs(
    id: str,
    rate_hz: float | None,
    offset_us: int,
    kind: ChannelKind,
    name: str = "",
    unit: str = "",
    offset_uv: float = 0.0,
) -> dict[str, object]:
    """Return the channel-group zarr.json attributes for one channel.

    offset_us is microseconds from the bundle's onset to this channel's
    first sample or event, never a wall-clock value. rate_hz is omitted when
    a channel has no sample rate, which an annotation channel does not.

    offset_uv is the DC offset removed from the statistics before folding. It
    is written only when there is one, since the attribute is optional and a
    reader that does not find it adds nothing back. Other values pass through
    unchanged.
    """
    attributes: dict[str, object] = {
        "id": id,
        "offset_us": offset_us,
        "kind": kind,
        "name": name,
        "unit": unit,
    }
    if rate_hz is not None:
        attributes["rate_hz"] = rate_hz
    if offset_uv:
        attributes["offset_uv"] = offset_uv
    return attributes


def annotation_channel_attrs(
    id: str,
    offset_us: int,
    name: str,
    unit: str,
    body_media_type: str | None,
    max_duration_us: int | None,
    label_names: list[str] | None,
) -> dict[str, object]:
    """Return the channel-group attributes for an annotation event channel.

    No rate_hz: marks are placed by their own timestamps, not sampled. Each
    optional key is written only when its column is there, since a reader
    decides what it can draw from which columns a channel carries.

    max_duration_us is what makes a window query find an interval that began
    before it and has not closed. Without it a reader would need an interval
    index; with it, it widens its backward search by one number.
    """
    attributes = channel_group_attrs(id, None, offset_us, "event", name, unit)
    if body_media_type is not None:
        attributes["body_media_type"] = body_media_type
    if max_duration_us is not None:
        attributes["max_duration_us"] = max_duration_us
    if label_names:
        attributes["label_names"] = label_names
    return attributes


def meta_group_attrs(
    subject: dict[str, object],
    session: dict[str, object],
    source: dict[str, object],
) -> dict[str, object]:
    """Return the meta/ group attributes: the bundle's whole identity surface.

    The three objects are deliberately unschematized; ontology belongs to
    archival standards. session carries start_us, the only wall-clock value
    anywhere in a bundle, which is what makes date-shifting a one-field edit.
    """
    return {"subject": subject, "session": session, "source": source}


def level_group_attrs(period_us: float) -> dict[str, object]:
    """Return the pyramid-level group attributes.

    period_us is the microseconds one bin spans at this level, and the only
    attribute a level group carries. It sits on the group rather than on a
    member array because every member of a level shares one bin axis.
    """
    return {"period_us": period_us}


def waveform_array_attrs(period_us: float) -> dict[str, object]:
    """Return the waveforms array attributes for a unit channel.

    period_us is the sample period within a spike waveform, not a
    pyramid-level period.
    """
    return {"period_us": period_us}
