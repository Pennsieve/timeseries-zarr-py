"""Build the attribute dicts written into the bundle's zarr.json files.

docs/spec.md specifies the keys and their types. Every 1.0 builder
returns its object under the single tsz key.
"""

import json

from timeseries_zarr.constants import FORMAT_VERSION, TSZ_KEY


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

    offset_us is the time of sample 0 in microseconds from the bundle origin.
    provenance is written only when it is not None. Raises ValueError if
    rate_hz is not positive, offset_us is negative, or provenance is not a
    JSON object of finite values.
    """
    if rate_hz <= 0:
        raise ValueError(f"rate_hz must be positive, got {rate_hz!r}")
    if offset_us < 0:
        raise ValueError(f"offset_us must be at least 0, got {offset_us!r}")
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


def level_group_attrs(period_us: float) -> dict[str, object]:
    """Return the level-group attributes.

    period_us is the microseconds one bin spans at this level. Raises
    ValueError if it is not positive.
    """
    if period_us <= 0:
        raise ValueError(f"period_us must be positive, got {period_us!r}")
    return _tsz({"period_us": period_us})


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
