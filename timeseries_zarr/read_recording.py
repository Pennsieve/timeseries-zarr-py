"""The parsed recording.json of a bundle, and its parsers."""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from types import MappingProxyType

_EMPTY: Mapping[str, object] = MappingProxyType({})

_LEAP_SECOND = re.compile(r"([T ]\d{2}:\d{2}:)60(?=[.Z+-]|$)")
"""Second 60 of an uppercase RFC 3339 time.

Group 1 holds the date separator, the hours, and the minutes before it.
"""

_JSON_TYPE_NAMES: Mapping[type, str] = {
    list: "an array",
    str: "a string",
    int: "a number",
    float: "a number",
    bool: "a boolean",
    type(None): "null",
}
"""JSON names of the Python types that json.loads returns for a non-object."""


@dataclass(frozen=True, slots=True)
class Recording:
    """Identity and datetimes read from recording.json.

    reference_time is the reference time, t = 0 of every time in the bundle,
    as a datetime. session_start_time is the start_time of the session
    object, the instant acquisition began. Each carries the UTC
    offset the file gives it and is None when the file omits it. subject,
    session, and source are producer-defined objects, read-only after parsing
    and empty when the file omits them. session keeps its start_time entry.
    """

    reference_time: datetime | None = None
    session_start_time: datetime | None = None
    subject: Mapping[str, object] = field(default=_EMPTY, hash=False)
    session: Mapping[str, object] = field(default=_EMPTY, hash=False)
    source: Mapping[str, object] = field(default=_EMPTY, hash=False)


def _optional_time(
    document: Mapping[str, object], key: str, name: str | None = None
) -> datetime | None:
    """Return the RFC 3339 time under key, or None when key is absent.

    name is how an error names the key, and defaults to key. The letters T
    and Z may be lowercase. A leap second, second 60, reads as the first
    instant of the next minute. The result drops any fraction finer than a
    microsecond. Raises ValueError if the value is not a string, does not
    parse, lies past the year 9999, or has no UTC offset.
    """
    if key not in document:
        return None
    value = document[key]
    key = key if name is None else name
    if not isinstance(value, str):
        raise ValueError(
            f"recording.json: {key} must be a string, got {value!r}"
        )
    try:
        parsed = _parse_time(value.upper())
    except OverflowError as error:
        raise ValueError(
            f"recording.json: {key} must lie in the year 9999 or earlier, "
            f"got {value!r}"
        ) from error
    except ValueError as error:
        raise ValueError(
            f"recording.json: {key} must be an RFC 3339 time, got {value!r}"
        ) from error
    if parsed.utcoffset() is None:
        raise ValueError(f"recording.json: {key} must carry a UTC offset")
    return parsed


def _parse_time(value: str) -> datetime:
    """Return the time that the uppercase string value names.

    Second 60 reads as the first instant of the next minute. Raises
    ValueError if value does not parse. Raises OverflowError if value is
    second 60 of the last minute of the year 9999.
    """
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        leap = _LEAP_SECOND.sub(r"\g<1>59", value, count=1)
        if leap == value:
            raise
        # datetime has no second 60, so second 59 plus 1 s gives the instant.
        return datetime.fromisoformat(leap) + timedelta(seconds=1)


def _optional_object(
    document: Mapping[str, object], key: str
) -> Mapping[str, object]:
    """Return a read-only copy of the object under key, empty when absent.

    Raises ValueError if the value is not an object.
    """
    if key not in document:
        return _EMPTY
    value = document[key]
    if not isinstance(value, Mapping):
        raise ValueError(
            f"recording.json: {key} must be an object, got {value!r}"
        )
    return MappingProxyType(dict(value))


def parse_recording_document(document: Mapping[str, object]) -> Recording:
    """Return the Recording that a parsed recording.json document describes.

    parse_recording_document reads only reference_time, subject, session,
    source, and the start_time of session. Raises ValueError if a time is not an RFC
    3339 string with a UTC offset, if a time lies past the year 9999, or if an
    object key holds a non-object.
    """
    session = _optional_object(document, "session")
    return Recording(
        reference_time=_optional_time(document, "reference_time"),
        session_start_time=_optional_time(
            session, "start_time", "session.start_time"
        ),
        subject=_optional_object(document, "subject"),
        session=session,
        source=_optional_object(document, "source"),
    )


def parse_recording_text(text: str) -> Recording:
    """Return the Recording that the text of a recording.json file describes.

    Raises ValueError if text is not valid JSON, if it holds a value other
    than an object, or if parse_recording_document rejects the object.
    """
    try:
        document = json.loads(text)
    except ValueError as error:
        raise ValueError(
            f"recording.json is not valid JSON: {error}"
        ) from error
    if not isinstance(document, Mapping):
        name = _JSON_TYPE_NAMES[type(document)]
        raise ValueError(f"recording.json must hold a JSON object, got {name}")
    return parse_recording_document(document)
