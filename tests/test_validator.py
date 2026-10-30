import json
import shutil

import numpy as np
import pytest
import zarr
from zarr.storage import StorePath, WrapperStore

from tests.conftest import (
    memory_store,
    write_mixed_bundle,
    write_pair_bundle,
    write_runs_bundle,
)
from timeseries_zarr import Finding, validate_bundle
from timeseries_zarr.validator import _is_datetime

HUGE = 2**53


def _read(path):
    return json.loads(path.read_text())


def _write(path, document):
    path.write_text(json.dumps(document))


def _node_file(bundle, path):
    return bundle / path / "zarr.json" if path else bundle / "zarr.json"


def edit(bundle, path, change):
    """Apply change to a zarr.json and to its consolidated entry."""
    file = _node_file(bundle, path)
    node = _read(file)
    change(node)
    _write(file, node)
    if path:
        root = _read(bundle / "zarr.json")
        change(root["consolidated_metadata"]["metadata"][path])
        _write(bundle / "zarr.json", root)


def edit_file_only(bundle, path, change):
    """Apply change to a zarr.json and leave the consolidated entry alone."""
    file = _node_file(bundle, path)
    node = _read(file)
    change(node)
    _write(file, node)


def edit_entry_only(bundle, path, change):
    """Apply change to a consolidated entry and leave the zarr.json alone."""
    root = _read(bundle / "zarr.json")
    change(root["consolidated_metadata"]["metadata"][path])
    _write(bundle / "zarr.json", root)


def edit_root(bundle, change):
    root = _read(bundle / "zarr.json")
    change(root)
    _write(bundle / "zarr.json", root)


def tsz(**fields):
    """Return a change that sets keys of the tsz object. None deletes a key."""

    def change(node):
        target = node["attributes"]["tsz"]
        for key, value in fields.items():
            if value is None:
                del target[key]
            else:
                target[key] = value

    return change


def top(**fields):
    """Return a change that sets keys of the node itself."""

    def change(node):
        node.update(fields)

    return change


def inner_codecs(change):
    """Return a change that edits the inner codec list of a sharded array."""

    def apply(node):
        change(node["codecs"][0]["configuration"])

    return apply


def remove(bundle, path):
    """Delete a node, its descendants, and their consolidated entries."""
    shutil.rmtree(bundle / path)
    root = _read(bundle / "zarr.json")
    metadata = root["consolidated_metadata"]["metadata"]
    for key in [k for k in metadata if k == path or k.startswith(path + "/")]:
        del metadata[key]
    _write(bundle / "zarr.json", root)


def copy_node(bundle, source, target):
    """Copy a node and its descendants, with their consolidated entries."""
    shutil.copytree(bundle / source, bundle / target)
    root = _read(bundle / "zarr.json")
    metadata = root["consolidated_metadata"]["metadata"]
    for key in [
        k for k in metadata if k == source or k.startswith(source + "/")
    ]:
        metadata[target + key[len(source) :]] = metadata[key]
    _write(bundle / "zarr.json", root)


def rename(bundle, source, target):
    copy_node(bundle, source, target)
    remove(bundle, source)


def write_array(bundle, path, index, value):
    """Assign value at index of the array path."""
    zarr.open_array(bundle / path, mode="r+")[index] = value


def drop_chunks(bundle, path):
    shutil.rmtree(bundle / path / "c")


@pytest.fixture
def bundle(full_bundle, tmp_path):
    destination = tmp_path / "bundle"
    shutil.copytree(full_bundle, destination)
    return destination


def _summary(findings):
    return [(f.path, f.section, f.message) for f in findings]


def assert_findings(findings, *expected):
    """Check that the findings are exactly expected, in any order.

    Each expected item is (path, section, a fragment of the message).
    """
    assert len(findings) == len(expected), _summary(findings)
    for path, section, fragment in expected:
        assert any(
            f.path == path and f.section == section and fragment in f.message
            for f in findings
        ), (path, section, fragment, _summary(findings))


def assert_contains(findings, path, section, fragment):
    assert any(
        f.path == path and f.section == section and fragment in f.message
        for f in findings
    ), (path, section, fragment, _summary(findings))


# Conforming bundles.


def test_the_full_bundle_has_no_findings(full_bundle):
    assert validate_bundle(full_bundle, full=True) == []


def test_a_finding_is_a_frozen_record():
    finding = Finding("0", "4.1", "message")
    assert (finding.path, finding.section, finding.message) == (
        "0",
        "4.1",
        "message",
    )
    with pytest.raises(AttributeError):
        finding.path = "1"


@pytest.mark.parametrize(
    "write", [write_mixed_bundle, write_pair_bundle, write_runs_bundle]
)
def test_a_writer_built_bundle_has_no_findings(write, tmp_path):
    assert validate_bundle(write(tmp_path / "bundle"), full=True) == []


def test_a_memory_store_and_a_store_path_validate(mixed_bundle):
    store = memory_store(mixed_bundle, ("", "sub/bundle"))
    assert validate_bundle(store, full=True) == []
    assert validate_bundle(StorePath(store, "sub/bundle")) == []


class _NoListing(WrapperStore):
    """A store that cannot list a prefix, like a web server."""

    async def list_dir(self, prefix):
        raise NotImplementedError
        yield


def test_a_store_that_cannot_list_validates_from_the_consolidated_metadata(
    full_bundle,
):
    store = _NoListing(memory_store(full_bundle))
    assert validate_bundle(store, full=True) == []


def test_a_file_url_validates(full_bundle):
    assert validate_bundle(full_bundle.as_uri(), full=True) == []


def test_a_missing_bundle_has_one_finding(tmp_path):
    assert_findings(
        validate_bundle(tmp_path / "absent"), ("", "3.1", "does not exist")
    )


def test_an_unsharded_chain_is_accepted(bundle):
    def plain(node):
        node["chunk_grid"]["configuration"]["chunk_shape"] = node["shape"]
        node["codecs"] = [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "zstd", "configuration": {"level": 3, "checksum": False}},
        ]

    edit(bundle, "0/1/mean", plain)
    assert validate_bundle(bundle) == []


def test_a_uint8_bytes_codec_may_omit_endian(bundle):
    node = _read(bundle / "2" / "text" / "zarr.json")
    inner = node["codecs"][0]["configuration"]["codecs"][0]
    assert inner == {"name": "bytes"}
    assert validate_bundle(bundle) == []


def test_missing_chunks_of_raw_and_levels_are_allowed(bundle):
    for path in ("0/raw", "0/1/count", "2/1/count"):
        drop_chunks(bundle, path)
    assert validate_bundle(bundle) == []


# Section 8.3, the zarr.json files and the consolidated metadata.


def test_a_root_without_a_zarr_json_stops_validation(bundle):
    _node_file(bundle, "").unlink()
    assert_findings(validate_bundle(bundle), ("", "3.1", "no zarr.json"))


def test_a_zarr_json_that_is_not_json_is_a_finding(bundle):
    _node_file(bundle, "0/1").write_text("{")
    assert_contains(validate_bundle(bundle), "0/1", "7.1", "not valid JSON")


def test_a_zarr_json_with_a_bare_nan_is_a_finding(bundle):
    text = (
        _node_file(bundle, "0/2/mean")
        .read_text()
        .replace('"zarr_format"', '"x": NaN, "zarr_format"')
    )
    _node_file(bundle, "0/2/mean").write_text(text)
    assert_contains(validate_bundle(bundle), "0/2/mean", "7.1", "NaN")


def test_a_group_with_zarr_format_2_is_a_finding(bundle):
    edit(bundle, "0/2", top(zarr_format=2))
    assert_findings(validate_bundle(bundle), ("0/2", "7.1", "zarr_format"))


def test_a_root_with_zarr_format_2_is_a_finding(bundle):
    edit_root(bundle, top(zarr_format=2))
    assert_findings(validate_bundle(bundle), ("", "7.1", "zarr_format"))


@pytest.mark.parametrize(
    ("version", "section", "fragment"),
    [
        ("2.0", "9", "major version 2"),
        ("1.1", "9", "minor version 1 is unsupported"),
        ("1", "6.2", "major.minor"),
        ("01.0", "6.2", "major.minor"),
        (1.0, "6.2", "major.minor"),
    ],
)
def test_the_root_version_is_checked(bundle, version, section, fragment):
    edit_root(bundle, tsz(version=version))
    assert_findings(validate_bundle(bundle), ("", section, fragment))


def test_a_root_without_tsz_is_a_finding(bundle):
    edit_root(bundle, top(attributes={}))
    assert_findings(validate_bundle(bundle), ("", "6.1", "tsz"))


def test_a_channel_group_with_a_version_is_a_finding(bundle):
    edit(bundle, "0", tsz(version="1.0"))
    assert_findings(validate_bundle(bundle), ("0", "6.2", "version"))


def test_a_root_without_consolidated_metadata_is_a_finding(bundle):
    edit_root(bundle, lambda node: node.pop("consolidated_metadata"))
    assert_findings(
        validate_bundle(bundle), ("", "6.2", "no consolidated_metadata")
    )


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        (lambda block: block.update(kind="external"), "kind"),
        (lambda block: block.update(must_understand=True), "must_understand"),
        (lambda block: block.pop("metadata"), "metadata"),
    ],
)
def test_the_consolidated_block_is_checked(bundle, change, fragment):
    edit_root(bundle, lambda node: change(node["consolidated_metadata"]))
    assert_findings(validate_bundle(bundle), ("", "6.2", fragment))


def test_a_node_missing_from_the_consolidated_metadata_is_a_finding(bundle):
    edit_root(
        bundle, lambda n: n["consolidated_metadata"]["metadata"].pop("0/1")
    )
    assert_findings(
        validate_bundle(bundle), ("0/1", "6.2", "not in the consolidated")
    )


def test_a_consolidated_entry_for_a_missing_node_is_a_finding(bundle):
    def add(node):
        metadata = node["consolidated_metadata"]["metadata"]
        metadata["9"] = metadata["0/1"]

    edit_root(bundle, add)
    assert_findings(
        validate_bundle(bundle), ("9", "6.2", "lists a missing node")
    )


def test_a_consolidated_entry_that_differs_is_a_finding(bundle):
    edit_entry_only(bundle, "0/1", tsz(period_us=1))
    assert_findings(validate_bundle(bundle), ("0/1", "6.2", "differs"))


def test_a_nested_consolidated_block_in_an_entry_is_ignored(bundle):
    edit_entry_only(
        bundle,
        "0",
        top(consolidated_metadata={"kind": "inline", "metadata": {"x": 1}}),
    )
    assert validate_bundle(bundle) == []


# Section 8.3, channel groups and their attributes.


def test_a_gap_in_the_channel_names_is_a_finding(bundle):
    rename(bundle, "2", "3")
    assert_findings(validate_bundle(bundle), ("", "3.2", "0 through C - 1"))


def test_an_array_with_an_integer_name_is_a_finding(bundle):
    copy_node(bundle, "0/raw", "3")
    assert_findings(validate_bundle(bundle), ("3", "3.2", "an array"))


def test_a_name_with_a_leading_zero_is_not_a_channel(bundle):
    copy_node(bundle, "2", "01")
    assert validate_bundle(bundle) == []


@pytest.mark.parametrize(
    ("path", "change", "section", "fragment"),
    [
        ("0", tsz(kind="spectral"), "6.3", "tsz.kind"),
        ("0", tsz(kind=None), "6.3", "tsz.kind"),
        ("0", tsz(name=None), "6.3", "tsz.name is required"),
        ("0", tsz(name=7), "6.3", "tsz.name must be a string"),
        ("0", tsz(id=7), "6.3", "tsz.id must be a string"),
        ("0", tsz(unit=None), "6.3", "tsz.unit is required"),
        ("0", tsz(rate_hz=0), "6.3", "tsz.rate_hz"),
        ("0", tsz(rate_hz="512"), "6.3", "tsz.rate_hz"),
        ("0", tsz(rate_hz=True), "6.3", "tsz.rate_hz"),
        ("0", tsz(rate_hz=None), "6.3", "tsz.rate_hz"),
        ("0", tsz(provenance=[1]), "6.3", "tsz.provenance"),
        ("2", tsz(unit=3), "6.3", "tsz.unit must be a string"),
        ("2", tsz(label_names=[]), "5.3", "at least 1 string"),
        ("2", tsz(label_names=["a", 1]), "5.3", "at least 1 string"),
        ("2", tsz(label_names="abc"), "5.3", "at least 1 string"),
        ("2", top(attributes={}), "6.1", "tsz"),
        ("2", top(attributes={"tsz": []}), "6.1", "tsz"),
    ],
)
def test_channel_attributes_are_checked(
    bundle, path, change, section, fragment
):
    edit(bundle, path, change)
    assert_findings(validate_bundle(bundle), (path, section, fragment))


# Section 8.3, time bounds.


def test_an_offset_written_as_a_float_is_a_finding(bundle):
    edit(bundle, "0", tsz(offset_us=5.0))
    assert_findings(
        validate_bundle(bundle),
        ("0", "2.3", "tsz.offset_us must be a JSON integer"),
    )


@pytest.mark.parametrize("value", [HUGE, -HUGE, "0", True])
def test_an_offset_outside_the_time_range_is_a_finding(bundle, value):
    edit(bundle, "0", tsz(offset_us=value))
    assert_findings(validate_bundle(bundle), ("0", "2.3", "tsz.offset_us"))


def test_a_grid_time_past_the_range_is_a_finding(bundle):
    remove(bundle, "0/runs")
    edit(bundle, "0", tsz(offset_us=HUGE - 100))
    assert_findings(
        validate_bundle(bundle), ("0", "2.3", "grid time of sample 1000")
    )


def test_a_max_duration_written_as_a_float_is_a_finding(bundle):
    edit(bundle, "2", tsz(max_duration_us=6_000_000.0))
    assert_findings(validate_bundle(bundle), ("2", "6.3", "max_duration_us"))


def test_a_negative_max_duration_is_a_finding(bundle):
    edit(bundle, "2", tsz(max_duration_us=-1))
    assert_findings(validate_bundle(bundle), ("2", "6.3", "max_duration_us"))


def test_a_max_duration_beyond_the_range_is_a_finding(bundle):
    edit(bundle, "2", tsz(max_duration_us=HUGE))
    assert_findings(validate_bundle(bundle), ("2", "6.3", "max_duration_us"))


def test_a_count_level_period_written_as_a_float_is_a_finding(bundle):
    edit(bundle, "2/1", tsz(period_us=1_024_000.0))
    assert_findings(validate_bundle(bundle), ("2/1", "5.7", "tsz.period_us"))


def test_the_end_of_the_last_count_bin_past_the_range_is_a_finding(bundle):
    write_array(bundle, "2/events", -1, HUGE - 5)
    findings = validate_bundle(bundle)
    assert_contains(findings, "2/1", "2.3", "end of the last bin")


def test_an_event_time_beyond_the_range_is_a_finding(bundle):
    write_array(bundle, "2/events", -1, HUGE)
    assert_contains(validate_bundle(bundle), "2/events", "2.3", "event time")


# Section 8.3, levels.


def test_a_gap_in_the_level_names_is_a_finding(bundle):
    remove(bundle, "0/1")
    assert_findings(validate_bundle(bundle), ("0", "4.3", "named ['2']"))


def test_a_gap_in_the_count_level_names_is_a_finding(bundle):
    remove(bundle, "2/1")
    assert_findings(validate_bundle(bundle), ("2", "5.7", "named ['2']"))


def test_a_level_without_a_period_is_a_finding(bundle):
    edit(bundle, "0/1", tsz(period_us=None))
    assert_findings(validate_bundle(bundle), ("0/1", "6.4", "tsz.period_us"))


@pytest.mark.parametrize("value", ["8", 0, -5, True])
def test_a_level_period_that_is_not_a_positive_number_is_a_finding(
    bundle, value
):
    edit(bundle, "0/1", tsz(period_us=value))
    assert_findings(validate_bundle(bundle), ("0/1", "6.4", "tsz.period_us"))


def test_a_continuous_level_with_an_offset_is_a_finding(bundle):
    edit(bundle, "0/1", tsz(offset_us=0))
    assert_findings(validate_bundle(bundle), ("0/1", "6.4", "offset_us"))


def test_a_level_period_off_by_more_than_the_tolerance_is_a_finding(bundle):
    period = _read(bundle / "0" / "1" / "zarr.json")["attributes"]["tsz"][
        "period_us"
    ]
    edit(bundle, "0/1", tsz(period_us=period * (1 + 1e-9)))
    assert_findings(validate_bundle(bundle), ("0/1", "4.3", "within 1e-12"))


def test_a_level_period_within_the_tolerance_is_accepted(bundle):
    period = _read(bundle / "0" / "1" / "zarr.json")["attributes"]["tsz"][
        "period_us"
    ]
    edit(bundle, "0/1", tsz(period_us=period * (1 + 1e-13)))
    assert validate_bundle(bundle) == []


def test_a_level_narrower_than_two_samples_is_a_finding(bundle):
    edit(bundle, "0/1", tsz(period_us=1e6 / 512))
    assert_findings(validate_bundle(bundle), ("0/1", "4.3", "at least 2"))


def test_a_level_as_wide_as_2_to_the_32_samples_is_a_finding(bundle):
    edit(bundle, "0/2", tsz(period_us=2**32 * 1e6 / 512))
    findings = validate_bundle(bundle)
    assert_contains(findings, "0/2", "4.3", "below 2^32")


def test_a_level_width_that_does_not_nest_is_a_finding(bundle):
    edit(bundle, "0/2", tsz(period_us=6 * 1e6 / 512))
    findings = validate_bundle(bundle)
    assert_contains(findings, "0/2", "4.3", "integer multiple")


def test_a_level_width_below_twice_the_previous_is_a_finding(bundle):
    edit(bundle, "0/2", tsz(period_us=4 * 1e6 / 512))
    findings = validate_bundle(bundle)
    assert_contains(findings, "0/2", "4.3", "at least 2 times")


def test_a_level_three_times_as_wide_as_the_previous_nests(bundle):
    # Level 2 of 1001 samples at width 12 has 84 rows, not the 63 stored.
    edit(bundle, "0/2", tsz(period_us=12 * 1e6 / 512))
    findings = validate_bundle(bundle)
    assert not any(f.section == "4.3" for f in findings)


def test_a_count_level_period_that_does_not_nest_is_a_finding(bundle):
    edit(bundle, "2/2", tsz(period_us=1_500_000))
    findings = validate_bundle(bundle)
    assert_contains(findings, "2/2", "5.7", "integer multiple")


def test_a_count_level_offset_that_is_not_a_multiple_is_a_finding(bundle):
    edit(bundle, "2/1", tsz(offset_us=500))
    assert_findings(validate_bundle(bundle), ("2/1", "6.4", "multiple"))


def test_a_count_level_offset_after_the_first_event_is_a_finding(bundle):
    edit(bundle, "2/1", tsz(offset_us=2_048_000))
    findings = validate_bundle(bundle)
    assert_contains(findings, "2/1", "6.4", "at most events[0]")


def test_a_count_level_needs_an_offset_when_events_start_below_0(bundle):
    write_array(bundle, "2/events", 0, -1)
    edit(bundle, "2/1", tsz(offset_us=None))
    edit(bundle, "2/2", tsz(offset_us=None))
    findings = validate_bundle(bundle)
    assert_contains(findings, "2/1", "6.4", "required")
    assert_contains(findings, "2/2", "6.4", "required")


def test_a_count_level_may_omit_the_offset_when_events_start_at_0_or_above(
    bundle,
):
    edit(bundle, "2/1", tsz(offset_us=None))
    assert validate_bundle(bundle) == []


def test_a_count_level_with_the_wrong_bin_count_is_a_finding(bundle):
    edit(bundle, "2/1/count", top(shape=[10, 3]))
    assert_findings(validate_bundle(bundle), ("2/1/count", "5.1", "shape"))


# Section 8.3, arrays.


def test_a_missing_raw_array_is_a_finding(bundle):
    remove(bundle, "0/raw")
    assert_findings(validate_bundle(bundle), ("0", "4.1", "raw is required"))


def test_a_missing_level_member_is_a_finding(bundle):
    remove(bundle, "0/1/count")
    assert_findings(
        validate_bundle(bundle), ("0/1", "4.1", "count is required")
    )


def test_a_missing_events_array_is_a_finding(bundle):
    remove(bundle, "2/events")
    findings = validate_bundle(bundle)
    assert_contains(findings, "2", "5.1", "events is required")


def test_a_missing_count_array_is_a_finding(bundle):
    remove(bundle, "2/2/count")
    assert_findings(
        validate_bundle(bundle), ("2/2", "5.1", "count is required")
    )


def test_a_node_that_should_be_an_array_is_a_finding(bundle):
    shutil.rmtree(bundle / "0" / "raw")
    copy_node(bundle, "0/1", "0/raw")
    findings = validate_bundle(bundle)
    assert_contains(findings, "0/raw", "4.1", "must be an array")


@pytest.mark.parametrize(
    ("path", "change", "section", "fragment"),
    [
        ("0/raw", top(data_type="float64"), "4.1", "data_type"),
        ("0/raw", top(fill_value=0), "7.3", "fill_value"),
        ("0/raw", top(fill_value="nan"), "7.3", "fill_value"),
        ("0/1/count", top(fill_value=1), "7.3", "fill_value"),
        ("0/1/count", top(fill_value=0.0), "7.3", "fill_value"),
        ("2/events", top(fill_value="NaN"), "7.3", "fill_value"),
        ("0/raw", top(shape=[10, 2]), "4.1", "shape"),
        ("0/1/mean", top(shape=[250]), "4.1", "shape"),
        ("0/1/minmax", top(shape=[251]), "4.1", "shape"),
        ("2/channels", top(shape=[40, 2]), "5.1", "shape"),
        ("2/labels", top(shape=[41]), "5.1", "shape"),
        ("0/raw", top(chunk_grid={"name": "irregular"}), "7.1", "chunk_grid"),
        (
            "0/raw",
            top(
                chunk_grid={
                    "name": "regular",
                    "configuration": {"chunk_shape": [1, 1]},
                }
            ),
            "7.1",
            "chunk_grid",
        ),
        (
            "0/raw",
            top(
                chunk_key_encoding={
                    "name": "default",
                    "configuration": {"separator": "."},
                }
            ),
            "7.1",
            "chunk_key_encoding",
        ),
        (
            "0/raw",
            top(chunk_key_encoding={"name": "v2"}),
            "7.1",
            "chunk_key_encoding",
        ),
        ("0/raw", top(codecs=[{"name": "blosc"}]), "7.2", "codec"),
        ("0/raw", top(codecs=[]), "7.2", "codec"),
        (
            "0/raw",
            inner_codecs(
                lambda config: config.update(
                    codecs=[{"name": "bytes"}, {"name": "gzip"}]
                )
            ),
            "7.2",
            "bytes, then optionally zstd",
        ),
        (
            "0/raw",
            inner_codecs(
                lambda config: config.update(codecs=[{"name": "bytes"}])
            ),
            "7.2",
            'declare "endian"',
        ),
        (
            "2/events",
            inner_codecs(
                lambda config: config["codecs"][0].update(
                    configuration={"endian": "big"}
                )
            ),
            "7.2",
            "endian 'big'",
        ),
        (
            "2/text",
            inner_codecs(
                lambda config: config["codecs"][0].update(
                    configuration={"endian": "big"}
                )
            ),
            "7.2",
            "endian 'big'",
        ),
        (
            "0/raw",
            inner_codecs(lambda config: config.update(index_location="start")),
            "7.2",
            "index location",
        ),
        (
            "0/raw",
            inner_codecs(
                lambda config: config.update(index_codecs=[{"name": "bytes"}])
            ),
            "7.2",
            "index codecs",
        ),
        (
            "0/raw",
            inner_codecs(
                lambda config: config["index_codecs"][0].update(
                    configuration={"endian": "big"}
                )
            ),
            "7.2",
            "index bytes codec",
        ),
        (
            "0/raw",
            inner_codecs(lambda config: config.update(chunk_shape=[3000])),
            "7.2",
            "inner chunk shape",
        ),
    ],
)
def test_array_metadata_is_checked(bundle, path, change, section, fragment):
    edit(bundle, path, change)
    assert_findings(validate_bundle(bundle), (path, section, fragment))


def test_a_text_width_of_0_is_a_finding(bundle):
    edit(bundle, "2/text", top(shape=[40, 0]))
    assert_findings(validate_bundle(bundle), ("2/text", "5.5", "W must be"))


def test_a_waveform_length_of_0_is_a_finding(bundle):
    edit(bundle, "2/waveforms", top(shape=[40, 0]))
    findings = validate_bundle(bundle)
    assert_contains(findings, "2/waveforms", "5.6", "P must be at least 1")


def test_a_channel_with_no_samples_and_a_level_is_a_finding(bundle):
    copy_node(bundle, "0/1", "1/1")
    findings = validate_bundle(bundle)
    assert_contains(findings, "1", "4.5", "no levels")


def test_a_channel_with_no_samples_and_runs_is_a_finding(bundle):
    copy_node(bundle, "0/runs", "1/runs")
    assert_findings(validate_bundle(bundle), ("1/runs", "4.5", "no runs"))


def test_an_unknown_array_and_group_are_ignored(bundle):
    copy_node(bundle, "0/raw", "0/extra")
    copy_node(bundle, "0/1", "0/notes")
    copy_node(bundle, "0/raw", "extra")
    copy_node(bundle, "0/1", "montages")
    assert validate_bundle(bundle) == []


def test_a_tsz_object_on_a_plain_array_is_a_finding(bundle):
    edit(bundle, "0/raw", top(attributes={"tsz": {}}))
    assert_findings(validate_bundle(bundle), ("0/raw", "6.1", "tsz"))


def test_other_attributes_beside_tsz_are_accepted(bundle):
    edit(bundle, "0", lambda node: node["attributes"].update(other={"a": 1}))
    edit(bundle, "0/raw", top(attributes={"units": "uV"}))
    assert validate_bundle(bundle) == []


# Section 8.3, runs.


def test_a_runs_table_that_breaks_section_4_2_is_a_finding(bundle):
    write_array(bundle, "0/runs", (0, 0), 1)
    assert_findings(validate_bundle(bundle), ("0/runs", "4.2", "row 0"))


def test_a_run_time_beyond_the_range_is_a_finding(bundle):
    write_array(bundle, "0/runs", (1, 2), HUGE)
    assert_findings(validate_bundle(bundle), ("0/runs", "2.3", "column 2"))


def test_a_run_that_ends_before_the_last_sample_is_a_finding(bundle):
    write_array(bundle, "0/runs", (1, 1), 400)
    assert_findings(validate_bundle(bundle), ("0/runs", "4.2", "last run"))


def test_a_run_that_starts_too_early_is_a_finding(bundle):
    write_array(bundle, "0/runs", 1, [400, 601, 781_249])
    assert_findings(validate_bundle(bundle), ("0/runs", "4.2", "before"))


# Section 8.3, events and columns.


def test_decreasing_event_times_are_a_finding(bundle):
    write_array(bundle, "2/events", 5, 0)
    assert_findings(
        validate_bundle(bundle), ("2/events", "5.1", "non-decreasing")
    )


def test_a_negative_duration_is_a_finding(bundle):
    write_array(bundle, "2/durations", 5, -1)
    assert_findings(
        validate_bundle(bundle), ("2/durations", "5.2", "at least 0")
    )


def test_a_duration_above_the_bound_is_a_finding(bundle):
    write_array(bundle, "2/durations", 5, 7_000_000)
    assert_findings(
        validate_bundle(bundle),
        ("2/durations", "5.2", "exceeds max_duration_us"),
    )


def test_a_duration_at_the_bound_is_accepted(bundle):
    write_array(bundle, "2/durations", 5, 6_000_000)
    assert validate_bundle(bundle) == []


def test_an_interval_that_ends_past_the_range_is_a_finding(bundle):
    edit(bundle, "2", tsz(max_duration_us=HUGE - 1))
    write_array(bundle, "2/durations", 5, HUGE - 1)
    assert_findings(
        validate_bundle(bundle), ("2/durations", "2.3", "end of an interval")
    )


def test_a_duration_beyond_the_range_is_a_finding(bundle):
    edit(bundle, "2", tsz(max_duration_us=HUGE - 1))
    write_array(bundle, "2/durations", 5, HUGE)
    findings = validate_bundle(bundle)
    assert_contains(findings, "2/durations", "2.3", "a duration lies outside")


def test_durations_without_a_bound_are_a_finding(bundle):
    edit(bundle, "2", tsz(max_duration_us=None))
    assert_findings(validate_bundle(bundle), ("2", "5.2", "required"))


def test_a_bound_without_durations_is_accepted(bundle):
    remove(bundle, "2/durations")
    assert validate_bundle(bundle) == []


def test_a_label_at_or_above_the_label_count_is_a_finding(bundle):
    write_array(bundle, "2/labels", 0, 3)
    assert_findings(
        validate_bundle(bundle), ("2/labels", "5.3", "label count 3")
    )


def test_the_label_count_without_names_is_one_above_the_largest_label(bundle):
    edit(bundle, "2", tsz(label_names=None))
    write_array(bundle, "2/labels", 0, 2)
    assert validate_bundle(bundle) == []


def test_waveform_mean_with_the_wrong_row_count_is_a_finding(bundle):
    edit(bundle, "2/waveform_mean", top(shape=[4, 8]))
    assert_findings(
        validate_bundle(bundle), ("2/waveform_mean", "5.1", "shape")
    )


def test_waveform_mean_without_labels_is_a_finding(bundle):
    remove(bundle, "2/labels")
    assert_findings(
        validate_bundle(bundle), ("2/waveform_mean", "5.6", "labels")
    )


@pytest.mark.parametrize(
    ("path", "change", "fragment"),
    [
        ("2/waveforms", tsz(period_us=0), "tsz.period_us"),
        ("2/waveform_mean", tsz(period_us="x"), "tsz.period_us"),
        ("2/waveforms", tsz(period_us=None), "tsz.period_us"),
        ("2/waveforms", tsz(event_sample=8), "tsz.event_sample"),
        ("2/waveforms", tsz(event_sample=-1), "tsz.event_sample"),
        ("2/waveforms", tsz(event_sample=1.5), "tsz.event_sample"),
        ("2/waveform_mean", tsz(event_sample=True), "tsz.event_sample"),
    ],
)
def test_waveform_attributes_are_checked(bundle, path, change, fragment):
    edit(bundle, path, change)
    assert_findings(validate_bundle(bundle), (path, "6.5", fragment))


def test_a_waveform_array_without_tsz_is_a_finding(bundle):
    edit(bundle, "2/waveforms", top(attributes={}))
    assert_findings(validate_bundle(bundle), ("2/waveforms", "6.1", "tsz"))


@pytest.mark.parametrize(
    "change",
    [tsz(period_us=1.0), tsz(event_sample=2)],
)
def test_waveforms_and_waveform_mean_must_agree(bundle, change):
    edit(bundle, "2/waveform_mean", change)
    assert_findings(
        validate_bundle(bundle), ("2/waveform_mean", "5.6", "must agree")
    )


def test_waveform_mean_without_waveforms_needs_no_agreement(bundle):
    remove(bundle, "2/waveforms")
    edit(bundle, "2/waveform_mean", tsz(period_us=1.0))
    assert validate_bundle(bundle, full=True) == []


def test_a_text_row_with_a_byte_after_a_zero_is_a_finding(bundle):
    write_array(bundle, "2/text", 2, np.frombuffer(b"a\0b\0\0\0", np.uint8))
    assert_findings(validate_bundle(bundle), ("2/text", "5.5", "row 2"))


def test_a_text_row_that_is_not_utf_8_is_a_finding(bundle):
    write_array(
        bundle, "2/text", 2, np.frombuffer(b"\xff\xfe\0\0\0\0", np.uint8)
    )
    assert_findings(validate_bundle(bundle), ("2/text", "5.5", "row 2"))


def test_a_text_row_that_is_cut_in_a_character_is_a_finding(bundle):
    cut = "caf\N{LATIN SMALL LETTER E WITH ACUTE}".encode()[:-1]
    write_array(bundle, "2/text", 2, np.frombuffer(cut + b"\0\0\0", np.uint8))
    assert_findings(validate_bundle(bundle), ("2/text", "5.5", "row 2"))


def test_a_text_row_that_fills_the_width_is_accepted(bundle):
    rows = np.asarray(zarr.open_array(bundle / "2" / "text", mode="r")[:])
    assert bytes(rows[3]) == b"abcdef"
    assert validate_bundle(bundle) == []


def test_a_channels_bit_at_or_above_the_channel_count_is_a_finding(bundle):
    write_array(bundle, "2/channels", (0, 0), 0b1000)
    assert_findings(validate_bundle(bundle), ("2/channels", "5.4", "3 or more"))


def test_every_channels_bit_below_the_channel_count_is_accepted(bundle):
    write_array(bundle, "2/channels", (0, 0), 0b111)
    assert validate_bundle(bundle) == []


# Section 8.3, chunk presence.


@pytest.mark.parametrize(
    "array",
    [
        "events",
        "durations",
        "labels",
        "values",
        "text",
        "channels",
        "waveforms",
        "waveform_mean",
    ],
)
def test_a_missing_event_column_chunk_is_a_finding(bundle, array):
    drop_chunks(bundle, f"2/{array}")
    assert_findings(
        validate_bundle(bundle), (f"2/{array}", "7.3", "chunks are missing")
    )


def test_a_missing_runs_chunk_is_a_finding(bundle):
    drop_chunks(bundle, "0/runs")
    assert_findings(
        validate_bundle(bundle), ("0/runs", "7.3", "chunks are missing")
    )


def test_an_unsharded_array_with_a_missing_chunk_is_a_finding(bundle):
    def plain(node):
        node["chunk_grid"]["configuration"]["chunk_shape"] = [16, 1]
        node["shape"] = [40, 1]
        node["codecs"] = [
            {"name": "bytes"},
            {"name": "zstd", "configuration": {"level": 1, "checksum": False}},
        ]

    edit(bundle, "2/channels", plain)
    findings = validate_bundle(bundle)
    assert_findings(findings, ("2/channels", "7.3", "2 chunks are missing"))


# Section 8.3, recording.json.


def test_recording_json_that_is_not_an_object_is_a_finding(bundle):
    (bundle / "recording.json").write_text("[]")
    assert_findings(
        validate_bundle(bundle), ("recording.json", "6.6", "object")
    )


def test_recording_json_that_is_not_json_is_a_finding(bundle):
    (bundle / "recording.json").write_text("{")
    assert_findings(validate_bundle(bundle), ("recording.json", "6.6", "JSON"))


@pytest.mark.parametrize(
    "document",
    [
        {"reference_time": "2024-03-14T14:15:00"},
        {"reference_time": 5},
        {"reference_time": "yesterday"},
        {"reference_time": None},
        {"session": {"start_time": None}},
        {"session": {"start_time": "2024-03-14 14:15:00+00:00"}},
        {"session": {"start_time": "2024-13-14T14:15:00Z"}},
    ],
)
def test_a_recording_datetime_that_is_not_rfc_3339_is_a_finding(
    bundle, document
):
    (bundle / "recording.json").write_text(json.dumps(document))
    assert_findings(
        validate_bundle(bundle), ("recording.json", "6.6", "RFC 3339")
    )


@pytest.mark.parametrize("key", ["subject", "session", "source"])
def test_a_recording_section_that_is_not_an_object_is_a_finding(bundle, key):
    (bundle / "recording.json").write_text(json.dumps({key: "x"}))
    assert_findings(validate_bundle(bundle), ("recording.json", "6.6", key))


def test_a_complete_recording_is_accepted(bundle):
    document = {
        "reference_time": "2024-03-14T09:15:00.123456-05:00",
        "subject": {"id": "s"},
        "session": {"start_time": "2024-03-14T14:15:00Z", "id": "x"},
        "source": {},
        "extra": 1,
    }
    (bundle / "recording.json").write_text(json.dumps(document))
    assert validate_bundle(bundle) == []


@pytest.mark.parametrize(
    "value",
    [
        "2024-03-14T14:15:00Z",
        "2024-03-14t14:15:00z",
        "2024-03-14T14:15:00.5+00:00",
        "2024-03-14T23:59:60Z",
        "2024-03-14T14:15:00-23:59",
    ],
)
def test_rfc_3339_datetimes_with_an_offset_are_recognized(value):
    assert _is_datetime(value)


@pytest.mark.parametrize(
    "value",
    [
        "2024-03-14T14:15:00",
        "2024-03-14",
        "2024-03-14T14:15Z",
        "2024-02-30T14:15:00Z",
        "2024-03-14T24:00:00Z",
        "2024-03-14T14:15:00+24:00",
        "2024-03-14T14:15:00.Z",
        20240314,
        None,
    ],
)
def test_other_values_are_not_rfc_3339_datetimes(value):
    assert not _is_datetime(value)


# Section 8.4.


def test_full_validation_reports_a_wrong_mean(bundle):
    write_array(bundle, "0/1/mean", 0, 5000.0)
    assert validate_bundle(bundle) == []
    assert_findings(
        validate_bundle(bundle, full=True), ("0/1/mean", "8.4", "bin 0")
    )


def test_full_validation_reports_a_wrong_variance(bundle):
    write_array(bundle, "0/1/var", 7, 1e6)
    assert_findings(
        validate_bundle(bundle, full=True), ("0/1/var", "8.4", "bin 7")
    )


def test_full_validation_reports_a_wrong_count(bundle):
    write_array(bundle, "0/1/count", 3, 9)
    assert_findings(
        validate_bundle(bundle, full=True),
        ("0/1/count", "8.4", "bin 3 holds 9"),
    )


def test_full_validation_reports_a_wrong_minmax(bundle):
    write_array(bundle, "0/2/minmax", (1, 1), 1e9)
    assert_findings(
        validate_bundle(bundle, full=True), ("0/2/minmax", "8.4", "bin 1")
    )


def test_full_validation_reports_a_non_nan_sample_outside_a_run(bundle):
    write_array(bundle, "0/raw", 450, 1.0)
    findings = validate_bundle(bundle, full=True)
    assert_contains(findings, "0/raw", "4.2", "sample 450")


def test_full_validation_reports_a_wrong_event_count(bundle):
    write_array(bundle, "2/1/count", (0, 0), 7)
    assert_findings(
        validate_bundle(bundle, full=True), ("2/1/count", "8.4", "bin 0")
    )


def test_full_validation_reports_a_wrong_waveform_mean(bundle):
    write_array(bundle, "2/waveform_mean", (1, 0), 3.0)
    assert_findings(
        validate_bundle(bundle, full=True), ("2/waveform_mean", "8.4", "row 1")
    )


def test_full_validation_reports_a_wrong_waveform_mean_of_an_empty_label(
    bundle,
):
    write_array(bundle, "2/waveform_mean", (1, 5), 0.0)
    assert_findings(
        validate_bundle(bundle, full=True), ("2/waveform_mean", "8.4", "row 1")
    )


def test_full_validation_reports_a_missing_events_chunk_once(bundle):
    drop_chunks(bundle, "2/events")
    assert_findings(
        validate_bundle(bundle, full=True),
        ("2/events", "7.3", "chunks are missing"),
    )


def test_full_validation_reports_a_missing_raw_chunk(bundle):
    drop_chunks(bundle, "0/raw")
    findings = validate_bundle(bundle, full=True)
    assert_contains(findings, "0/1/count", "8.4", "bin 0")


def test_full_validation_accepts_a_mean_within_the_tolerance(bundle):
    mean = zarr.open_array(bundle / "0" / "1" / "mean", mode="r")
    stored = np.float32(mean[0])
    write_array(bundle, "0/1/mean", 0, np.nextafter(stored, np.float32(1e9)))
    assert validate_bundle(bundle, full=True) == []


def test_full_validation_reads_nothing_after_a_structural_failure(bundle):
    edit(bundle, "0/1/mean", top(data_type="float64"))
    findings = validate_bundle(bundle, full=True)
    assert_findings(findings, ("0/1/mean", "4.1", "data_type"))
