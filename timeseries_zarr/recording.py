"""The bundle's recording.json: identity and wall-clock time in one plain file."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from timeseries_zarr.types import RecordingMeta

RECORDING_FILE = "recording.json"
"""Name of the plain JSON file at the root of a bundle."""

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def session_start_us(meta: RecordingMeta) -> int | None:
    """Return the session start in microseconds since the Unix epoch.

    The arithmetic is exact in integers. A meta without a start returns None.
    Raises ValueError if meta.start has no UTC offset.
    """
    if meta.start is None:
        return None
    if meta.start.utcoffset() is None:
        raise ValueError("meta.start must carry a UTC offset")
    return (meta.start - _EPOCH) // timedelta(microseconds=1)


def recording_document(
    meta: RecordingMeta, origin_us: int
) -> dict[str, object]:
    """Return the recording.json content for a bundle with the given origin.

    start is the bundle origin, the instant every time in the bundle counts
    from, as an RFC 3339 string with microsecond precision. It is written in
    the UTC offset of meta.start, and it comes from origin_us and not from
    meta.start, because a channel can begin before the session start. origin_us
    is wall-clock microseconds since the Unix epoch. A meta without a start
    gives a document without one, and origin_us is then unused. subject,
    session, and source are copied in that order and only when they hold
    something. Raises ValueError if meta.start has no UTC offset.
    """
    document: dict[str, object] = {}
    if meta.start is not None:
        if meta.start.utcoffset() is None:
            raise ValueError("meta.start must carry a UTC offset")
        origin = (_EPOCH + timedelta(microseconds=origin_us)).astimezone(
            meta.start.tzinfo
        )
        document["start"] = origin.isoformat(timespec="microseconds")
    for key, value in (
        ("subject", meta.subject),
        ("session", meta.session),
        ("source", meta.source),
    ):
        if value:
            document[key] = dict(value)
    return document


def write_recording_json(bundle_dir: Path, document: dict[str, object]) -> None:
    """Write document as recording.json at the root of bundle_dir.

    The file is UTF-8 JSON indented by one space and ends with one newline, so
    an editor or a text tool can change it. It is not a Zarr node and never
    enters the consolidated metadata. An existing file is overwritten.
    bundle_dir must exist. Raises TypeError if a value in document is not JSON
    serializable.
    """
    text = json.dumps(document, indent=1, ensure_ascii=False)
    (bundle_dir / RECORDING_FILE).write_text(text + "\n", encoding="utf-8")
