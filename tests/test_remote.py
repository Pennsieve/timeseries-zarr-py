import threading
import zipfile
from contextlib import contextmanager
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import override

import numpy as np
import pytest
from zarr.storage import FsspecStore

from timeseries_zarr.read_bundle import open_bundle
from timeseries_zarr.recording import RECORDING_FILE
from timeseries_zarr.zarr_io import read_store_text

fsspec = pytest.importorskip("fsspec")

_MEMORY_DIRECTORY = "test-remote"
_MEMORY_URL = f"memory://{_MEMORY_DIRECTORY}/bundle"
_MEMORY_LOCATION = f"memory:///{_MEMORY_DIRECTORY}/bundle"
_HEADER = "X-Tsz-Test"
_READ_LENGTHS = [512, 2048, 3]


@pytest.fixture(scope="module")
def mixed(mixed_bundle):
    return open_bundle(mixed_bundle)


def _copy_to_memory(path, directory):
    """Copy every file under path to the fsspec memory filesystem.

    The files land under /directory/bundle at their paths relative to path.
    """
    # The trailing slash copies the contents even when the target exists.
    fsspec.filesystem("memory").put(
        f"{path}/", f"/{directory}/bundle", recursive=True
    )


@pytest.fixture(scope="module")
def memory_url(mixed_bundle):
    """Copy the mixed bundle to _MEMORY_URL and remove it after the module."""
    _copy_to_memory(mixed_bundle, _MEMORY_DIRECTORY)
    yield _MEMORY_URL
    fsspec.filesystem("memory").rm(f"/{_MEMORY_DIRECTORY}", recursive=True)


def _reads(bundle):
    """Return 2 s of raw samples, level 1 bins, and a window of events."""
    channel = bundle.continuous[0]
    return [
        channel.read(0, 2_000_000),
        channel.read(0, 32_000_000, level=1),
        bundle.events[3].read(3_000_000, 20_000_000),
    ]


def _assert_same_reads(bundle, mixed):
    """Assert that bundle reads the same values and recording as mixed."""
    reads = _reads(bundle)
    assert [len(result) for result in reads] == _READ_LENGTHS
    for result, expected in zip(reads, _reads(mixed), strict=True):
        columns = result.columns()
        expected_columns = expected.columns()
        assert list(columns) == list(expected_columns)
        for name, values in expected_columns.items():
            np.testing.assert_array_equal(columns[name], values)
    assert bundle.recording is not None
    assert bundle.recording == mixed.recording


def test_the_memory_copy_holds_each_bundle_file_at_its_relative_path(
    mixed_bundle, memory_url
):
    lead = f"/{_MEMORY_DIRECTORY}/bundle/"
    copied = {
        path.removeprefix(lead)
        for path in fsspec.filesystem("memory").find(f"/{_MEMORY_DIRECTORY}")
    }
    assert copied == {
        file.relative_to(mixed_bundle).as_posix()
        for file in mixed_bundle.rglob("*")
        if file.is_file()
    }


def test_open_bundle_reads_a_memory_url_as_it_reads_the_bundle_on_disk(
    mixed, memory_url
):
    bundle = open_bundle(memory_url)
    assert isinstance(bundle.root.store, FsspecStore)
    _assert_same_reads(bundle, mixed)


def test_location_of_a_memory_url_bundle_is_its_url_and_reopens(memory_url):
    bundle = open_bundle(memory_url)
    assert bundle.location == _MEMORY_LOCATION
    reopened = open_bundle(bundle.location)
    assert list(reopened.channels) == list(bundle.channels)


def test_location_of_a_chained_memory_url_bundle_is_its_chain_and_reopens(
    memory_url,
):
    bundle = open_bundle(f"simplecache::{memory_url}")
    assert bundle.location == f"simplecache::{_MEMORY_LOCATION}"
    reopened = open_bundle(bundle.location)
    assert list(reopened.channels) == list(bundle.channels)


def test_location_of_a_zip_archive_bundle_is_its_url_in_the_archive(
    mixed, mixed_bundle, tmp_path
):
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as writer:
        for file in sorted(mixed_bundle.rglob("*")):
            if file.is_file():
                writer.write(file, file.relative_to(mixed_bundle).as_posix())
    bundle = open_bundle(f"zip://::{archive.as_uri()}")
    _assert_same_reads(bundle, mixed)
    assert bundle.location == "zip://"


def test_open_bundle_rejects_storage_options_with_a_memory_url(memory_url):
    with pytest.raises(TypeError, match="storage_options"):
        open_bundle(memory_url, storage_options={"skip_instance_cache": True})


def test_open_bundle_reads_an_fsspec_store_made_with_storage_options(
    mixed, memory_url
):
    store = FsspecStore.from_url(
        memory_url,
        storage_options={"skip_instance_cache": True},
        read_only=True,
    )
    bundle = open_bundle(store)
    _assert_same_reads(bundle, mixed)
    assert bundle.location == _MEMORY_LOCATION


def test_open_bundle_of_a_memory_url_without_recording_json_has_no_recording(
    mixed_bundle,
):
    directory = "test-remote-bare"
    _copy_to_memory(mixed_bundle, directory)
    filesystem = fsspec.filesystem("memory")
    try:
        filesystem.rm(f"/{directory}/bundle/{RECORDING_FILE}")
        bundle = open_bundle(f"memory://{directory}/bundle")
        assert bundle.recording is None
        assert len(bundle.channels) == 5
    finally:
        filesystem.rm(f"/{directory}", recursive=True)


def test_close_closes_a_memory_url_bundle_and_a_second_call_succeeds(
    memory_url,
):
    bundle = open_bundle(memory_url)
    bundle.close()
    bundle.close()
    assert not bundle.root.store._is_open


class _QuietRequestHandler(SimpleHTTPRequestHandler):
    """Serve each whole file and ignore a Range header."""

    @override
    def log_message(self, format, *args):
        """Keep the request log out of the test output."""


class _RangeRequestHandler(_QuietRequestHandler):
    """Serve each file, or the byte range that a Range header names.

    Records the method, the path, and the _HEADER value of each GET and HEAD
    in server.requests.
    """

    def _record_request(self):
        self.server.requests.append(
            (self.command, self.path, self.headers.get(_HEADER))
        )

    @override
    def do_HEAD(self):
        self._record_request()
        super().do_HEAD()

    @override
    def do_GET(self):
        self._record_request()
        file = Path(self.translate_path(self.path))
        if not file.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = file.read_bytes()
        status = HTTPStatus.OK
        requested = self.headers.get("Range")
        if requested is not None:
            first, _, last = requested.removeprefix("bytes=").partition("-")
            if first:
                data = data[int(first) : int(last) + 1 if last else None]
            else:
                data = data[-int(last) :]
            status = HTTPStatus.PARTIAL_CONTENT
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@contextmanager
def _serving(handler_class, directory):
    """Serve directory on 127.0.0.1 with handler_class until the block ends."""
    handler = partial(handler_class, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture(scope="module")
def http_server(mixed_bundle):
    """Serve the parent directory of the mixed bundle on 127.0.0.1."""
    pytest.importorskip("aiohttp")
    with _serving(_RangeRequestHandler, mixed_bundle.parent) as server:
        yield server


def _server_url(server, name):
    host, port = server.server_address[:2]
    return f"http://{host}:{port}/{name}"


@pytest.fixture
def http_url(http_server, mixed_bundle):
    """Return the URL of the mixed bundle on a server with no requests."""
    http_server.requests.clear()
    return _server_url(http_server, mixed_bundle.name)


def test_open_bundle_reads_an_http_url_as_it_reads_the_bundle_on_disk(
    mixed, http_url
):
    bundle = open_bundle(http_url)
    assert isinstance(bundle.root.store, FsspecStore)
    _assert_same_reads(bundle, mixed)


def test_location_of_an_http_url_bundle_is_its_url(http_url):
    assert open_bundle(http_url).location == http_url


def test_open_bundle_over_http_fetches_only_zarr_json_and_recording_json(
    http_server, http_url, mixed_bundle
):
    open_bundle(http_url)
    assert [request[:2] for request in http_server.requests] == [
        ("GET", f"/{mixed_bundle.name}/zarr.json"),
        ("GET", f"/{mixed_bundle.name}/{RECORDING_FILE}"),
    ]


def test_open_bundle_sends_the_storage_options_with_each_http_request(
    http_server, http_url
):
    bundle = open_bundle(http_url, storage_options={"headers": {_HEADER: "1"}})
    bundle.continuous[0].read(0, 1_000_000)
    bundle.events[3].read()
    assert len(http_server.requests) > 2
    assert {value for _, _, value in http_server.requests} == {"1"}


def test_level_read_of_no_other_member_over_http_fetches_only_minmax(
    http_server, http_url
):
    channel = open_bundle(http_url).continuous[0]
    http_server.requests.clear()
    channel.read(0, 32_000_000, level=1, members=())
    paths = [path for _, path, _ in http_server.requests]
    assert any("/1/minmax/" in path for path in paths)
    for name in ("mean", "var", "count"):
        assert not any(f"/1/{name}/" in path for path in paths)


def test_read_store_text_returns_none_for_an_object_the_server_lacks(
    http_url,
):
    root = open_bundle(http_url).root
    assert read_store_text(root, "absent.json") is None


def test_open_bundle_raises_file_not_found_for_an_http_url_without_a_bundle(
    http_server,
):
    with pytest.raises(FileNotFoundError):
        open_bundle(_server_url(http_server, "absent"))


@pytest.fixture(scope="module")
def no_range_url(mixed_bundle):
    """Return the URL of the mixed bundle on a server that ignores Range."""
    pytest.importorskip("aiohttp")
    with _serving(_QuietRequestHandler, mixed_bundle.parent) as server:
        yield _server_url(server, mixed_bundle.name)


def test_open_bundle_succeeds_over_a_server_that_ignores_range(no_range_url):
    bundle = open_bundle(no_range_url)
    assert list(bundle.channels) == [0, 1, 2, 3, 4]


def test_a_read_of_some_inner_chunks_of_a_shard_raises_without_range(
    no_range_url,
):
    channel = open_bundle(no_range_url).continuous[0]
    # One sample needs one of the four inner chunks of shard 0.
    with pytest.raises(ValueError):
        channel.read(0, 1000)


def test_a_read_of_every_inner_chunk_of_a_shard_succeeds_without_range(
    mixed, no_range_url
):
    channel = open_bundle(no_range_url).continuous[0]
    # 1000 samples at 256 Hz end in the last of the four inner chunks of
    # shard 0.
    result = channel.read(0, 3_906_000)
    assert len(result) == 1000
    np.testing.assert_array_equal(
        result.values, mixed.continuous[0].read(0, 3_906_000).values
    )
