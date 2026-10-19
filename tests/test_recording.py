import json
from datetime import UTC, datetime, timedelta, timezone

import pytest

from timeseries_zarr.recording import (
    RECORDING_FILE,
    epoch_us,
    recording_document,
    write_recording_json,
)
from timeseries_zarr.types import RecordingMeta

_NOON_UTC_US = 1_710_407_700 * 1_000_000
"""2024-03-14T09:15:00 UTC in microseconds since the epoch."""
_UTC_MINUS_5 = timezone(timedelta(hours=-5))


def _meta(start=None, subject=None, session=None, source=None, **options):
    return RecordingMeta(
        session_start_time=start or datetime(2024, 3, 14, 9, 15, tzinfo=UTC),
        subject={} if subject is None else subject,
        session={} if session is None else session,
        source={} if source is None else source,
        **options,
    )


def test_recording_document_start_time_is_the_recording_start_in_utc():
    doc = recording_document(_meta(), _NOON_UTC_US)
    assert doc["start_time"] == "2024-03-14T09:15:00.000000+00:00"


def test_recording_document_start_time_keeps_the_offset_of_the_session_start():
    start = datetime(2024, 3, 14, 4, 15, tzinfo=_UTC_MINUS_5)
    doc = recording_document(_meta(start=start), _NOON_UTC_US)
    assert doc["start_time"] == "2024-03-14T04:15:00.000000-05:00"


def test_recording_document_start_time_is_in_utc_without_a_session_start():
    doc = recording_document(RecordingMeta(), _NOON_UTC_US)
    assert doc == {"start_time": "2024-03-14T09:15:00.000000+00:00"}


def test_recording_document_start_time_is_exact_to_the_microsecond():
    doc = recording_document(_meta(), 1_700_000_000_123_457)
    assert doc["start_time"] == "2023-11-14T22:13:20.123457+00:00"


def test_recording_document_start_time_parses_back_to_the_recording_start():
    doc = recording_document(_meta(), _NOON_UTC_US + 5)
    parsed = datetime.fromisoformat(doc["start_time"])
    assert parsed.utcoffset() is not None
    assert parsed == datetime(2024, 3, 14, 9, 15, 0, 5, tzinfo=UTC)


def test_recording_document_writes_the_session_start_as_given():
    start = datetime(2024, 3, 14, 4, 15, 0, 7, tzinfo=_UTC_MINUS_5)
    doc = recording_document(_meta(start=start), _NOON_UTC_US - 90 * 1_000_000)
    assert doc["start_time"] == "2024-03-14T04:13:30.000000-05:00"
    assert doc["session"] == {"start_time": "2024-03-14T04:15:00.000007-05:00"}


def test_recording_document_puts_the_session_start_before_the_other_session_keys():
    doc = recording_document(_meta(session={"id": "ses-01"}), _NOON_UTC_US)
    assert list(doc["session"]) == ["start_time", "id"]


def test_recording_document_times_are_equal_when_the_recording_starts_with_the_session():
    doc = recording_document(_meta(), _NOON_UTC_US)
    assert doc["session"]["start_time"] == doc["start_time"]


def test_recording_document_without_a_recording_start_has_no_start_time():
    doc = recording_document(_meta(subject={"id": "sub-0931"}), None)
    assert list(doc) == ["subject", "session"]
    assert doc["session"] == {"start_time": "2024-03-14T09:15:00.000000+00:00"}


def test_recording_document_without_wall_clock_times_has_no_start_time():
    doc = recording_document(_meta(wall_clock=False), _NOON_UTC_US)
    assert list(doc) == ["session"]


def test_recording_document_rejects_a_session_start_without_a_utc_offset():
    with pytest.raises(ValueError, match="session_start_time must carry a UTC"):
        recording_document(_meta(start=datetime(2024, 3, 14, 9, 15)), 0)


def test_recording_document_rejects_a_session_object_that_holds_a_start_time():
    meta = _meta(session={"start_time": "2024-03-14T09:15:00+00:00"})
    with pytest.raises(ValueError, match="session holds a start_time key"):
        recording_document(meta, _NOON_UTC_US)


def test_recording_document_omits_objects_that_are_empty():
    assert list(recording_document(RecordingMeta(), 0)) == ["start_time"]


def test_recording_document_of_an_empty_meta_without_a_start_is_empty():
    assert recording_document(RecordingMeta(), None) == {}


def test_recording_document_copies_the_objects_in_a_fixed_order():
    meta = _meta(
        subject={"id": "sub-0931", "sex": "F"},
        session={"id": "ses-01"},
        source={"system": "Natus"},
    )
    doc = recording_document(meta, _NOON_UTC_US)
    assert list(doc) == ["start_time", "subject", "session", "source"]
    assert doc["subject"] == {"id": "sub-0931", "sex": "F"}
    assert doc["session"] == {
        "start_time": "2024-03-14T09:15:00.000000+00:00",
        "id": "ses-01",
    }
    assert doc["source"] == {"system": "Natus"}


def test_recording_document_does_not_alias_the_metadata():
    meta = _meta(subject={"id": "sub-0931"}, session={"id": "ses-01"})
    doc = recording_document(meta, 0)
    doc["subject"]["id"] = "changed"
    doc["session"]["id"] = "changed"
    assert meta.subject == {"id": "sub-0931"}
    assert meta.session == {"id": "ses-01"}


def test_write_recording_json_round_trips_the_document(tmp_path):
    doc = recording_document(
        _meta(subject={"id": "sub-0931"}, session={"id": "ses-01"}),
        _NOON_UTC_US,
    )
    write_recording_json(tmp_path, doc)
    assert json.loads((tmp_path / RECORDING_FILE).read_text("utf-8")) == doc


def test_write_recording_json_is_indented_utf8_with_one_trailing_newline(
    tmp_path,
):
    write_recording_json(tmp_path, {"time": "x", "session": {"id": "a"}})
    text = (tmp_path / RECORDING_FILE).read_text("utf-8")
    assert text == '{\n "time": "x",\n "session": {\n  "id": "a"\n }\n}\n'


def test_write_recording_json_keeps_non_ascii_text_readable(tmp_path):
    word = "caf\N{LATIN SMALL LETTER E WITH ACUTE}"
    write_recording_json(tmp_path, {"subject": {"note": word}})
    raw = (tmp_path / RECORDING_FILE).read_bytes()
    assert word.encode("utf-8") in raw


def test_write_recording_json_overwrites_an_existing_file(tmp_path):
    write_recording_json(tmp_path, {"time": "old"})
    write_recording_json(tmp_path, {"time": "new"})
    stored = json.loads((tmp_path / RECORDING_FILE).read_text("utf-8"))
    assert stored == {"time": "new"}


def test_write_recording_json_creates_no_zarr_node(tmp_path):
    write_recording_json(tmp_path, {"time": "x"})
    assert [p.name for p in tmp_path.iterdir()] == [RECORDING_FILE]


def test_write_recording_json_requires_an_existing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        write_recording_json(tmp_path / "missing", {"time": "x"})


def test_write_recording_json_rejects_a_value_that_is_not_json(tmp_path):
    with pytest.raises(TypeError):
        write_recording_json(tmp_path, {"subject": {"tags": {1, 2}}})


def test_epoch_us_is_exact_microseconds_since_the_epoch():
    start = datetime(2023, 11, 14, 22, 13, 20, 123457, tzinfo=UTC)
    assert epoch_us(start) == 1_700_000_000_123_457


def test_epoch_us_is_the_same_instant_in_any_offset():
    start = datetime(2024, 3, 14, 4, 15, tzinfo=_UTC_MINUS_5)
    assert epoch_us(start) == _NOON_UTC_US


def test_epoch_us_rejects_an_instant_without_a_utc_offset():
    with pytest.raises(ValueError, match="instant must carry a UTC offset"):
        epoch_us(datetime(2024, 3, 14, 9, 15))
