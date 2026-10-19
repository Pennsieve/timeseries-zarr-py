"""Build and parse the attributes of the bundle's zarr.json files.

docs/spec.md specifies the keys and their types. Every 1.0 builder
returns its object under the single tsz key. Each parser names its node in
its error messages. parse_version names the root group, and every other
parser takes a where string such as "channel 0 level 2". A parser reads only
the keys it names.
"""

import copy
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from timeseries_zarr.constants import (
    CHANNEL_KINDS,
    FORMAT_VERSION,
    MAJOR_VERSION,
    TSZ_KEY,
)
from timeseries_zarr.types import ChannelKind

_VERSION_PATTERN: re.Pattern[str] = re.compile(r"([0-9]+)\.[0-9]+")

_STRING = "a string"
_VERSION = "a string of the form major.minor"
_POSITIVE_NUMBER = "a finite number above 0"
_NONNEGATIVE_INTEGER = "an integer from 0 to 2^63 - 1"
_INT64_MAX = 2**63 - 1
_STRING_LIST = "a non-empty list of strings"
_OBJECT = "an object"


@dataclass(frozen=True, slots=True)
class ContinuousAttrs:
    """Parsed tsz fields of a continuous channel group.

    rate_hz is the sample rate in hertz. offset_us is the time of sample 0
    in microseconds from the bundle origin. id and provenance are None when
    the group omits them.
    """

    name: str
    id: str | None
    unit: str
    rate_hz: float
    offset_us: int
    provenance: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class EventAttrs:
    """Parsed tsz fields of an event channel group.

    Every field but name is None when the group omits its key. label_names[j]
    names label value j. max_duration_us bounds every duration in
    microseconds.
    """

    name: str
    id: str | None
    unit: str | None
    label_names: tuple[str, ...] | None
    max_duration_us: int | None
    provenance: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class WaveformAttrs:
    """Parsed tsz fields of a waveforms or waveform_mean array.

    period_us is the sample period within a snippet in microseconds.
    event_sample is the snippet index recorded at the event time.
    """

    period_us: float
    event_sample: int


def _tsz(fields: dict[str, object]) -> dict[str, object]:
    """Return fields as the value of the tsz attribute key."""
    return {TSZ_KEY: fields}


def check_provenance(provenance: object) -> None:
    """Raise ValueError unless provenance is None or a storable JSON object.

    A non-finite number would be written as a bare NaN or Infinity, which is
    not JSON, so one such value makes the root metadata unreadable.
    """
    if provenance is None:
        return
    if not isinstance(provenance, dict):
        raise ValueError(
            "provenance must be a dict or None, "
            f"got {type(provenance).__name__}"
        )
    try:
        json.dumps(provenance, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"provenance must hold only JSON values with finite numbers: {error}"
        ) from error


def root_attrs() -> dict[str, object]:
    """Return the root group attributes: the format version under tsz."""
    return _tsz({"version": FORMAT_VERSION})


def continuous_channel_attrs(
    name: str,
    id: str,
    rate_hz: float,
    offset_us: int,
    unit: str,
    provenance: dict[str, object] | None,
) -> dict[str, object]:
    """Return the channel-group attributes for one continuous channel.

    offset_us is the time of sample 0 in microseconds from the reference
    time, and may be negative. provenance is written only when it is not None.
    Raises ValueError if rate_hz is not positive or provenance is not a JSON
    object of finite values.
    """
    if rate_hz <= 0:
        raise ValueError(f"rate_hz must be positive, got {rate_hz!r}")
    check_provenance(provenance)
    fields: dict[str, object] = {
        "kind": "continuous",
        "name": name,
        "id": id,
        "rate_hz": rate_hz,
        "offset_us": offset_us,
        "unit": unit,
    }
    if provenance is not None:
        fields["provenance"] = provenance
    return _tsz(fields)


def event_channel_attrs(
    name: str,
    id: str | None,
    unit: str | None,
    label_names: list[str] | None,
    max_duration_us: int | None,
    provenance: dict[str, object] | None,
) -> dict[str, object]:
    """Return the channel-group attributes for one event channel.

    kind and name are always present; each other key is written only when
    its argument is not None. Raises ValueError if label_names is empty,
    max_duration_us is negative, or provenance is not a JSON object of finite
    values.
    """
    if label_names is not None and not label_names:
        raise ValueError("label_names must name at least one category")
    if max_duration_us is not None and max_duration_us < 0:
        raise ValueError(
            f"max_duration_us must be at least 0, got {max_duration_us!r}"
        )
    check_provenance(provenance)
    fields: dict[str, object] = {"kind": "event", "name": name}
    optional: dict[str, object | None] = {
        "id": id,
        "unit": unit,
        "label_names": label_names,
        "max_duration_us": max_duration_us,
        "provenance": provenance,
    }
    fields.update(
        {key: value for key, value in optional.items() if value is not None}
    )
    return _tsz(fields)


def level_group_attrs(
    period_us: float, offset_us: int | None = None
) -> dict[str, object]:
    """Return the level-group attributes.

    period_us is the microseconds one bin spans at this level. offset_us,
    given for a count level, is the time of bin 0 from the reference time, a
    multiple of period_us. A level of a continuous channel passes None.
    Raises ValueError if period_us is not positive or offset_us is not a
    multiple of it.
    """
    if period_us <= 0:
        raise ValueError(f"period_us must be positive, got {period_us!r}")
    fields: dict[str, object] = {"period_us": period_us}
    if offset_us is not None:
        if offset_us % period_us:
            raise ValueError(
                f"offset_us must be a multiple of period_us, got {offset_us!r}"
                f" and {period_us!r}"
            )
        fields["offset_us"] = offset_us
    return _tsz(fields)


def waveform_array_attrs(
    period_us: float, event_sample: int | None = None
) -> dict[str, object]:
    """Return the attributes of a waveforms or waveform_mean array.

    period_us is the sample period within a snippet. event_sample is the
    snippet index recorded at the event time and is written only when given.
    Raises ValueError if period_us is not positive or event_sample is
    negative.
    """
    if period_us <= 0:
        raise ValueError(f"period_us must be positive, got {period_us!r}")
    if event_sample is not None and event_sample < 0:
        raise ValueError(
            f"event_sample must be at least 0, got {event_sample!r}"
        )
    fields: dict[str, object] = {"period_us": period_us}
    if event_sample is not None:
        fields["event_sample"] = event_sample
    return _tsz(fields)


def _invalid(
    where: str, key: str, expectation: str, value: object
) -> ValueError:
    """Return the error for a tsz key that holds a value it cannot hold."""
    return ValueError(
        f"{where}: {TSZ_KEY}.{key} must be {expectation}, got {value!r}"
    )


def _required(
    fields: Mapping[str, object], key: str, where: str, expectation: str
) -> object:
    """Return the value of a key the node must carry.

    Raises ValueError if the key is absent.
    """
    if key not in fields:
        raise ValueError(
            f"{where}: {TSZ_KEY}.{key} is missing; it must be {expectation}"
        )
    return fields[key]


def _as_finite_float(value: object) -> float | None:
    """Return value as a float if it is a finite number, else None.

    A bool is not a number here, although Python counts it as an int.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        number = float(value)
    except OverflowError:  # an int beyond the float range
        return None
    return number if math.isfinite(number) else None


def _as_integer(value: object) -> int | None:
    """Return value as an int if it is an integer, else None.

    An integral float such as 500000.0 counts as an integer. A bool does not.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _string(fields: Mapping[str, object], key: str, where: str) -> str:
    """Return the string a node must carry under key.

    Raises ValueError if the key is absent or its value is not a string.
    """
    value = _required(fields, key, where, _STRING)
    if not isinstance(value, str):
        raise _invalid(where, key, _STRING, value)
    return value


def _optional_string(
    fields: Mapping[str, object], key: str, where: str
) -> str | None:
    """Return the string under key, or None when the key is absent.

    Raises ValueError if the value is not a string.
    """
    if key not in fields:
        return None
    return _string(fields, key, where)


def _positive_number(
    fields: Mapping[str, object], key: str, where: str
) -> float:
    """Return the finite number above 0 a node must carry under key.

    Raises ValueError if the key is absent or its value is not such a number.
    """
    value = _required(fields, key, where, _POSITIVE_NUMBER)
    number = _as_finite_float(value)
    if number is None or number <= 0:
        raise _invalid(where, key, _POSITIVE_NUMBER, value)
    return number


def _optional_nonnegative_integer(
    fields: Mapping[str, object], key: str, where: str
) -> int | None:
    """Return the integer from 0 to 2^63 - 1 under key, or None when absent.

    Raises ValueError if the value is not such an integer.
    """
    if key not in fields:
        return None
    value = fields[key]
    integer = _as_integer(value)
    if integer is None or integer < 0 or integer > _INT64_MAX:
        raise _invalid(where, key, _NONNEGATIVE_INTEGER, value)
    return integer


def _nonnegative_integer(
    fields: Mapping[str, object], key: str, where: str, default: int
) -> int:
    """Return the integer from 0 to 2^63 - 1 under key, or default if absent.

    Raises ValueError if the value is not such an integer.
    """
    integer = _optional_nonnegative_integer(fields, key, where)
    return default if integer is None else integer


def _optional_string_tuple(
    fields: Mapping[str, object], key: str, where: str
) -> tuple[str, ...] | None:
    """Return the list of strings under key as a tuple, or None when absent.

    Raises ValueError if the value is not a non-empty list of strings.
    """
    if key not in fields:
        return None
    value = fields[key]
    if (
        not isinstance(value, list | tuple)
        or not value
        or not all(isinstance(item, str) for item in value)
    ):
        raise _invalid(where, key, _STRING_LIST, value)
    return tuple(value)


def _optional_object(
    fields: Mapping[str, object], key: str, where: str
) -> Mapping[str, object] | None:
    """Return a read-only copy of the object under key, or None when absent.

    The copy is deep, and the objects and lists nested in it are plain dicts
    and lists. Raises ValueError if the value is not an object.
    """
    if key not in fields:
        return None
    value = fields[key]
    if not isinstance(value, Mapping):
        raise _invalid(where, key, _OBJECT, value)
    return MappingProxyType(copy.deepcopy(dict(value)))


def tsz_fields(
    attributes: Mapping[str, object], where: str
) -> Mapping[str, object]:
    """Return the tsz object of a node's attributes.

    Raises ValueError if tsz is absent or is not an object.
    """
    if TSZ_KEY not in attributes:
        raise ValueError(f"{where}: {TSZ_KEY} is missing; it must be {_OBJECT}")
    value = attributes[TSZ_KEY]
    if not isinstance(value, Mapping):
        raise ValueError(f"{where}: {TSZ_KEY} must be {_OBJECT}, got {value!r}")
    return value


def parse_version(attributes: Mapping[str, object]) -> str:
    """Return the tsz.version of a root group whose major version is known.

    Raises ValueError unless the version is a major.minor string of decimal
    digits whose major is MAJOR_VERSION.
    """
    where = "root group"
    fields = tsz_fields(attributes, where)
    value = _required(fields, "version", where, _VERSION)
    if not isinstance(value, str):
        raise _invalid(where, "version", _VERSION, value)
    match = _VERSION_PATTERN.fullmatch(value)
    if match is None:
        raise _invalid(where, "version", _VERSION, value)
    major = int(match.group(1))
    if major != MAJOR_VERSION:
        raise ValueError(
            f"bundle version {value!r} has major version {major}; "
            f"this package reads major version {MAJOR_VERSION}"
        )
    return value


def parse_channel_kind(
    attributes: Mapping[str, object], where: str
) -> ChannelKind | None:
    """Return the tsz.kind of a channel group, or None for an unknown kind.

    A kind is known when CHANNEL_KINDS lists it. Raises ValueError if kind is
    absent or is not a string.
    """
    kind = _string(tsz_fields(attributes, where), "kind", where)
    for known in CHANNEL_KINDS:
        if kind == known:
            # The constant, unlike kind, carries the ChannelKind literal type.
            return known
    return None


def parse_continuous_attrs(
    attributes: Mapping[str, object], where: str
) -> ContinuousAttrs:
    """Return the tsz fields of a continuous channel group.

    offset_us defaults to 0. Raises ValueError if name, unit, or rate_hz is
    absent, or if a key holds a value outside its type or range.
    """
    fields = tsz_fields(attributes, where)
    return ContinuousAttrs(
        name=_string(fields, "name", where),
        id=_optional_string(fields, "id", where),
        unit=_string(fields, "unit", where),
        rate_hz=_positive_number(fields, "rate_hz", where),
        offset_us=_nonnegative_integer(fields, "offset_us", where, 0),
        provenance=_optional_object(fields, "provenance", where),
    )


def parse_event_attrs(
    attributes: Mapping[str, object], where: str
) -> EventAttrs:
    """Return the tsz fields of an event channel group.

    Raises ValueError if name is absent, or if a key holds a value outside
    its type or range.
    """
    fields = tsz_fields(attributes, where)
    return EventAttrs(
        name=_string(fields, "name", where),
        id=_optional_string(fields, "id", where),
        unit=_optional_string(fields, "unit", where),
        label_names=_optional_string_tuple(fields, "label_names", where),
        max_duration_us=_optional_nonnegative_integer(
            fields, "max_duration_us", where
        ),
        provenance=_optional_object(fields, "provenance", where),
    )


def parse_level_attrs(attributes: Mapping[str, object], where: str) -> float:
    """Return the bin width of a level group in microseconds.

    Raises ValueError unless tsz.period_us is a finite number above 0.
    """
    return _positive_number(tsz_fields(attributes, where), "period_us", where)


def parse_waveform_attrs(
    attributes: Mapping[str, object], where: str
) -> WaveformAttrs:
    """Return the tsz fields of a waveforms or waveform_mean array.

    event_sample defaults to 0. Raises ValueError if period_us is absent, or
    if a key holds a value outside its type or range.
    """
    fields = tsz_fields(attributes, where)
    return WaveformAttrs(
        period_us=_positive_number(fields, "period_us", where),
        event_sample=_nonnegative_integer(fields, "event_sample", where, 0),
    )
