import dataclasses
import json
from datetime import UTC, datetime, timedelta, timezone

import pytest

from timeseries_zarr.read_recording import (
    Recording,
    parse_recording_document,
    parse_recording_text,
)
from timeseries_zarr.recording import (
    RECORDING_FILE,
    recording_document,
    write_recording_json,
)
from timeseries_zarr.types import RecordingMeta

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_UTC_MINUS_5 = timezone(timedelta(hours=-5))
_SESSION = datetime(2024, 3, 14, 9, 15, 0, 7, tzinfo=_UTC_MINUS_5)
_SESSION_US = (_SESSION - _EPOCH) // timedelta(microseconds=1)

_TIME_KEYS = ["session_start_time", "timestamps_reference_time"]
_OBJECT_KEYS = ["subject", "session", "source"]


def test_recording_defaults_to_no_times_and_empty_objects():
    recording = Recording()
    assert recording.session_start_time is None
    assert recording.timestamps_reference_time is None
    assert recording.subject == {}
    assert recording.session == {}
    assert recording.source == {}


def test_recording_rejects_assignment():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Recording().subject = {}
    assert not hasattr(Recording(), "__dict__")


def test_equal_recordings_hash_equal():
    document = {
        "session_start_time": "2024-03-14T14:15:00Z",
        "subject": {"id": "sub-synth01", "age": "P30D"},
        "source": {"generator": "synthetic"},
    }
    first = parse_recording_document(document)
    second = parse_recording_document(json.loads(json.dumps(document)))
    assert first == second
    assert hash(first) == hash(second)
    assert len({first, second, Recording()}) == 2
    assert hash(Recording()) == hash(Recording())


def test_parse_recording_document_returns_the_times_recording_document_wrote():
    document = recording_document(
        RecordingMeta(session_start_time=_SESSION), _SESSION_US
    )
    recording = parse_recording_document(document)
    assert recording.session_start_time == _SESSION
    assert recording.timestamps_reference_time == _SESSION


def test_parse_recording_document_returns_an_origin_earlier_than_the_session():
    origin_us = _SESSION_US - 32_000
    document = recording_document(
        RecordingMeta(session_start_time=_SESSION), origin_us
    )
    recording = parse_recording_document(document)
    assert recording.session_start_time == _SESSION
    assert recording.timestamps_reference_time == _EPOCH + timedelta(
        microseconds=origin_us
    )
    assert recording.timestamps_reference_time < recording.session_start_time


def test_parse_recording_document_keeps_the_utc_offset_of_each_time():
    document = recording_document(
        RecordingMeta(session_start_time=_SESSION), _SESSION_US - 32_000
    )
    recording = parse_recording_document(document)
    assert recording.session_start_time is not None
    assert recording.timestamps_reference_time is not None
    assert recording.session_start_time.utcoffset() == timedelta(hours=-5)
    assert recording.timestamps_reference_time.utcoffset() == timedelta(
        hours=-5
    )


def test_parse_recording_document_returns_the_objects_recording_document_wrote():
    meta = RecordingMeta(
        session_start_time=_SESSION,
        subject={"id": "sub-0931", "age": "P34Y"},
        session={"id": "ses-01"},
        source={"system": "Natus Quantum"},
    )
    recording = parse_recording_document(recording_document(meta, _SESSION_US))
    assert recording.subject == {"id": "sub-0931", "age": "P34Y"}
    assert recording.session == {"id": "ses-01"}
    assert recording.source == {"system": "Natus Quantum"}


def test_parse_recording_text_reads_the_file_write_recording_json_writes(
    tmp_path,
):
    meta = RecordingMeta(
        session_start_time=_SESSION,
        subject={
            "id": "sub-0931",
            "name": "Jos\N{LATIN SMALL LETTER E WITH ACUTE}",
        },
    )
    write_recording_json(tmp_path, recording_document(meta, _SESSION_US))
    text = (tmp_path / RECORDING_FILE).read_text(encoding="utf-8")
    recording = parse_recording_text(text)
    assert recording.session_start_time == _SESSION
    assert recording.timestamps_reference_time == _SESSION
    assert recording.subject == meta.subject


def test_parse_recording_document_of_only_a_subject_has_no_times():
    recording = parse_recording_document({"subject": {"id": "sub-synth01"}})
    assert recording.session_start_time is None
    assert recording.timestamps_reference_time is None
    assert recording.subject == {"id": "sub-synth01"}
    assert recording.session == {}
    assert recording.source == {}


def test_parse_recording_document_of_an_empty_object_equals_the_default():
    assert parse_recording_document({}) == Recording()


def test_parse_recording_document_ignores_a_legacy_start_key():
    recording = parse_recording_document(
        {"start": "2024-03-14T09:15:00.000000-05:00", "subject": {"id": "s"}}
    )
    assert recording.session_start_time is None
    assert recording.timestamps_reference_time is None
    assert recording.subject == {"id": "s"}


def test_parse_recording_document_ignores_unknown_keys():
    recording = parse_recording_document(
        {"session_start_time": "2024-03-14T14:15:00Z", "notes": [1, 2]}
    )
    assert recording.session_start_time == datetime(
        2024, 3, 14, 14, 15, tzinfo=UTC
    )


@pytest.mark.parametrize("key", _TIME_KEYS)
def test_parse_recording_document_accepts_a_lowercase_t_and_z(key):
    recording = parse_recording_document({key: "2024-03-14t14:15:00z"})
    assert getattr(recording, key) == datetime(2024, 3, 14, 14, 15, tzinfo=UTC)


def test_parse_recording_document_reads_a_z_suffix_as_utc():
    recording = parse_recording_document(
        {"timestamps_reference_time": "2024-03-14T14:15:00.250000Z"}
    )
    time = recording.timestamps_reference_time
    assert time is not None
    assert time.utcoffset() == timedelta(0)
    assert time == datetime(2024, 3, 14, 14, 15, 0, 250_000, tzinfo=UTC)


def test_parse_recording_document_truncates_a_nanosecond_fraction():
    recording = parse_recording_document(
        {"session_start_time": "2024-03-14T14:15:00.123456789+00:00"}
    )
    assert recording.session_start_time == datetime(
        2024, 3, 14, 14, 15, 0, 123_456, tzinfo=UTC
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2016-12-31T23:59:60Z", datetime(2017, 1, 1, tzinfo=UTC)),
        (
            "2016-12-31t23:59:60.25z",
            datetime(2017, 1, 1, 0, 0, 0, 250_000, tzinfo=UTC),
        ),
        (
            "2016-12-31T18:59:60-05:00",
            datetime(2016, 12, 31, 19, tzinfo=_UTC_MINUS_5),
        ),
    ],
)
@pytest.mark.parametrize("key", _TIME_KEYS)
def test_parse_recording_document_reads_a_leap_second_as_the_next_minute(
    key, value, expected
):
    time = getattr(parse_recording_document({key: value}), key)
    assert time == expected
    assert time.utcoffset() == expected.utcoffset()


@pytest.mark.parametrize(
    "value", ["2016-12-31T23:59:61Z", "2016-13-31T23:59:60Z"]
)
def test_parse_recording_document_rejects_second_61_and_a_bad_leap_second(
    value,
):
    with pytest.raises(
        ValueError,
        match=(
            "^recording.json: session_start_time must be an RFC 3339 time, "
            f"got '{value}'$"
        ),
    ):
        parse_recording_document({"session_start_time": value})


def test_parse_recording_document_rejects_a_leap_second_past_the_year_9999():
    value = "9999-12-31T23:59:60Z"
    with pytest.raises(
        ValueError,
        match=(
            "^recording.json: timestamps_reference_time must lie in the year "
            f"9999 or earlier, got '{value}'$"
        ),
    ):
        parse_recording_document({"timestamps_reference_time": value})


@pytest.mark.parametrize("key", _TIME_KEYS)
def test_parse_recording_document_rejects_a_time_without_a_utc_offset(key):
    with pytest.raises(
        ValueError, match=f"^recording.json: {key} must carry a UTC offset$"
    ):
        parse_recording_document({key: "2024-03-14T14:15:00"})


def test_parse_recording_document_rejects_a_date_without_a_time():
    with pytest.raises(ValueError, match="session_start_time must carry"):
        parse_recording_document({"session_start_time": "2024-03-14"})


@pytest.mark.parametrize("key", _TIME_KEYS)
@pytest.mark.parametrize("value", [1_710_425_700, None, ["2024-03-14"]])
def test_parse_recording_document_rejects_a_time_that_is_not_a_string(
    key, value
):
    with pytest.raises(
        ValueError, match=f"^recording.json: {key} must be a string, got "
    ):
        parse_recording_document({key: value})


@pytest.mark.parametrize("key", _TIME_KEYS)
def test_parse_recording_document_rejects_a_string_that_is_not_a_time(key):
    with pytest.raises(
        ValueError,
        match=f"^recording.json: {key} must be an RFC 3339 time, got 'noon'$",
    ):
        parse_recording_document({key: "noon"})


@pytest.mark.parametrize("key", _OBJECT_KEYS)
@pytest.mark.parametrize("value", [["sub-0931"], "sub-0931", None])
def test_parse_recording_document_rejects_an_object_key_that_holds_a_non_object(
    key, value
):
    with pytest.raises(
        ValueError, match=f"^recording.json: {key} must be an object, got "
    ):
        parse_recording_document({key: value})


def test_parse_recording_document_copies_each_object():
    subject: dict[str, object] = {"id": "sub-0931"}
    recording = parse_recording_document({"subject": subject})
    subject["id"] = "changed"
    assert recording.subject == {"id": "sub-0931"}


@pytest.mark.parametrize("key", _OBJECT_KEYS)
def test_parse_recording_document_returns_read_only_objects(key):
    recording = parse_recording_document({key: {"id": "x"}})
    with pytest.raises(TypeError):
        getattr(recording, key)["id"] = "y"


def test_parse_recording_text_parses_a_json_object():
    text = json.dumps(
        {
            "session_start_time": "2024-03-14T09:15:00.000000-05:00",
            "timestamps_reference_time": "2024-03-14T09:14:59.968000-05:00",
            "session": {"id": "ses-01"},
        }
    )
    recording = parse_recording_text(text)
    assert recording.session_start_time == datetime(
        2024, 3, 14, 9, 15, tzinfo=_UTC_MINUS_5
    )
    assert recording.timestamps_reference_time == datetime(
        2024, 3, 14, 9, 14, 59, 968_000, tzinfo=_UTC_MINUS_5
    )
    assert recording.session == {"id": "ses-01"}


@pytest.mark.parametrize("text", ["{", "", "{'subject': {}}", "NaN-"])
def test_parse_recording_text_rejects_invalid_json(text):
    with pytest.raises(ValueError, match="^recording.json is not valid JSON: "):
        parse_recording_text(text)


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ('[{"subject": {}}]', "an array"),
        ('"2024-03-14"', "a string"),
        ("7", "a number"),
        ("7.5", "a number"),
        ("true", "a boolean"),
        ("null", "null"),
    ],
)
def test_parse_recording_text_rejects_a_document_that_is_not_an_object(
    text, name
):
    with pytest.raises(
        ValueError,
        match=f"^recording.json must hold a JSON object, got {name}$",
    ):
        parse_recording_text(text)


def test_parse_recording_text_applies_the_document_rules():
    with pytest.raises(ValueError, match="subject must be an object"):
        parse_recording_text('{"subject": 5}')
