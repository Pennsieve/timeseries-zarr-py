"""Time reads of a 64-channel bundle and the numcodecs Zstd decode.

The script writes a bundle to a temporary directory, then reports the median
time of Bundle.stack over the whole length and over 2000 samples at several
max_workers values, and of one-channel reads. It also times numcodecs Zstd
decode alone, to find out whether decode releases the GIL and whether decode
writes into a slice of a larger array without a copy.

Run it with `venv/bin/python scripts/bench_reads.py`. Every time is in
milliseconds and is the median of the repeats after one warm-up call.
"""

import os
import statistics
import tempfile
import time
import tracemalloc
import warnings
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import numpy.typing as npt
from numcodecs.zstd import Zstd

import timeseries_zarr as tsz

CHANNEL_COUNT = 64
SAMPLE_COUNT = 300_000
RATE_HZ = 30_000.0
WINDOW_SAMPLES = 30_000
WINDOW_START_SAMPLE = 150_000
WINDOW_SAMPLE_COUNT = 2000
LEVEL = 1
QUANTUM_UV = 0.25
STACK_WORKERS = (1, 8, 32)
DECODE_THREADS = (1, 32)
CHUNK_SAMPLES = 8192
ZSTD_LEVEL = 5
GIL_SPEEDUP_MINIMUM = 2.0
GIL_CORE_MINIMUM = 8
WHOLE_REPEATS = 5
WINDOW_REPEATS = 15
DECODE_REPEATS = 5
CHANNEL_REPEATS = 15
MICROSECONDS_PER_SECOND = 1_000_000

warnings.filterwarnings(
    "ignore",
    message="Consolidated metadata is currently not part",
    category=UserWarning,
)


@dataclass(frozen=True, slots=True)
class ChannelInfo:
    """Metadata of one benchmark channel."""

    id: str
    name: str
    unit: str

    def rate_hz(self) -> float:
        """Return the sample rate."""
        return RATE_HZ

    def start_us(self) -> int:
        """Return the start time."""
        return 0

    def num_samples(self) -> int:
        """Return the sample count."""
        return SAMPLE_COUNT


class BenchSource:
    """Every channel of the benchmark bundle, served in windows of rows."""

    def __init__(self, data: npt.NDArray[np.float32]) -> None:
        """Hold data, an array of (samples, channels)."""
        self._data = data
        self._channels = [
            ChannelInfo(f"ch{index}", f"channel {index}", "uV")
            for index in range(data.shape[1])
        ]

    def channels(self) -> list[ChannelInfo]:
        """Return the channels in channel order."""
        return self._channels

    def iter_windows(self) -> Iterator[npt.NDArray[np.float32]]:
        """Yield the samples as windows of (rows, channels)."""
        for start in range(0, self._data.shape[0], WINDOW_SAMPLES):
            yield self._data[start : start + WINDOW_SAMPLES]


def make_data() -> npt.NDArray[np.float32]:
    """Return quantized noise on a sine for every channel.

    The values step by QUANTUM_UV, as an analog-to-digital converter gives, so
    Zstd compresses them. The shape is (samples, channels).
    """
    rng = np.random.default_rng(0)
    time_s = np.arange(SAMPLE_COUNT) / RATE_HZ
    sine = 100.0 * np.sin(2 * np.pi * 10 * time_s)
    noise = rng.standard_normal((SAMPLE_COUNT, CHANNEL_COUNT)) * 20.0
    quantized = np.round((sine[:, None] + noise) / QUANTUM_UV) * QUANTUM_UV
    return quantized.astype(np.float32)


def median_ms(call: Callable[[], object], repeats: int) -> float:
    """Return the median wall time of call in milliseconds.

    Args:
        call: The work to time. One untimed call comes first.
        repeats: The number of timed calls.
    """
    call()
    times: list[float] = []
    for _ in range(repeats):
        begin = time.perf_counter()
        call()
        times.append(time.perf_counter() - begin)
    return statistics.median(times) * 1000.0


def sample_to_us(sample: int) -> int:
    """Return the time of a sample index in microseconds."""
    return round(sample * MICROSECONDS_PER_SECOND / RATE_HZ)


def bench_stack(bundle: tsz.Bundle) -> None:
    """Print the Bundle.stack times at each max_workers."""
    indices = list(range(CHANNEL_COUNT))
    start_us = sample_to_us(WINDOW_START_SAMPLE)
    end_us = sample_to_us(WINDOW_START_SAMPLE + WINDOW_SAMPLE_COUNT)
    for workers in STACK_WORKERS:
        whole = median_ms(
            partial(bundle.stack, indices, max_workers=workers),
            WHOLE_REPEATS,
        )
        print(
            f"stack raw whole length {CHANNEL_COUNT} channels "
            f"max_workers={workers}: {whole:.2f} ms"
        )
        window = median_ms(
            partial(
                bundle.stack, indices, start_us, end_us, max_workers=workers
            ),
            WINDOW_REPEATS,
        )
        print(
            f"stack raw {WINDOW_SAMPLE_COUNT} samples {CHANNEL_COUNT} channels "
            f"max_workers={workers}: {window:.2f} ms "
            f"({window / CHANNEL_COUNT:.3f} ms per channel)"
        )


def bench_channel(bundle: tsz.Bundle) -> None:
    """Print the times of one-channel reads of raw samples and of a level."""
    channel = bundle.continuous[0]
    start_us = sample_to_us(WINDOW_START_SAMPLE)
    end_us = sample_to_us(WINDOW_START_SAMPLE + WINDOW_SAMPLE_COUNT)
    cases: list[tuple[str, Callable[[], object]]] = [
        ("raw whole length", channel.read),
        (
            f"raw {WINDOW_SAMPLE_COUNT} samples",
            lambda: channel.read(start_us, end_us),
        ),
        (f"level {LEVEL} whole length", lambda: channel.read(level=LEVEL)),
        (
            f"level {LEVEL} {WINDOW_SAMPLE_COUNT} samples",
            lambda: channel.read(start_us, end_us, level=LEVEL),
        ),
    ]
    for label, call in cases:
        repeats = WHOLE_REPEATS if "whole" in label else CHANNEL_REPEATS
        print(f"channel read {label}: {median_ms(call, repeats):.3f} ms")


def decode_chunks(
    codec: Zstd, encoded: list[bytes], threads: int
) -> list[npt.NDArray[np.float32]]:
    """Decode every chunk in encoded on threads threads."""

    def decode(buf: bytes) -> npt.NDArray[np.float32]:
        return np.frombuffer(codec.decode(buf), dtype=np.float32)

    if threads == 1:
        return [decode(buf) for buf in encoded]
    with ThreadPoolExecutor(max_workers=threads) as pool:
        return list(pool.map(decode, encoded))


def bench_decode(data: npt.NDArray[np.float32]) -> tuple[int, float]:
    """Print the decode times and return the core count and the speedup.

    Args:
        data: Samples as (samples, channels).

    Returns:
        The CPU count and the 32-thread speedup over one thread.
    """
    channel_major = np.ascontiguousarray(data.T)
    chunks_per_channel = -(-SAMPLE_COUNT // CHUNK_SAMPLES)
    padded = np.zeros(
        (CHANNEL_COUNT, chunks_per_channel * CHUNK_SAMPLES), dtype=np.float32
    )
    padded[:, :SAMPLE_COUNT] = channel_major
    chunks = padded.reshape(-1, CHUNK_SAMPLES)
    codec = Zstd(level=ZSTD_LEVEL)
    encoded = [bytes(codec.encode(chunk)) for chunk in chunks]
    encoded_bytes = sum(len(buf) for buf in encoded)
    print(
        f"decode input: {len(encoded)} chunks of {CHUNK_SAMPLES} float32 "
        f"samples, {encoded_bytes} bytes encoded"
    )
    times: dict[int, float] = {}
    for threads in DECODE_THREADS:
        times[threads] = median_ms(
            partial(decode_chunks, codec, encoded, threads),
            DECODE_REPEATS,
        )
        print(
            f"zstd decode {len(encoded)} chunks threads={threads}: "
            f"{times[threads]:.2f} ms"
        )
    cores = os.cpu_count() or 1
    speedup = times[DECODE_THREADS[0]] / times[DECODE_THREADS[-1]]
    releases = speedup >= GIL_SPEEDUP_MINIMUM
    print(f"cores (os.cpu_count): {cores}")
    print(
        f"zstd decode speedup at {DECODE_THREADS[-1]} threads: {speedup:.2f}x"
    )
    if cores < GIL_CORE_MINIMUM:
        print(
            f"GIL verdict: not conclusive on fewer than {GIL_CORE_MINIMUM} cores"
        )
    else:
        print(
            "GIL verdict: numcodecs Zstd decode "
            + ("releases the GIL" if releases else "holds the GIL")
        )
    bench_decode_out(codec, encoded)
    return cores, speedup


def bench_decode_out(codec: Zstd, encoded: list[bytes]) -> None:
    """Print whether Zstd.decode(buf, out=...) writes in place without a copy.

    Args:
        codec: The codec that encoded encoded.
        encoded: Encoded chunks of CHUNK_SAMPLES float32 samples.
    """
    count = len(encoded)
    expected = [
        np.frombuffer(codec.decode(buf), dtype=np.float32) for buf in encoded
    ]
    big = np.zeros(count * CHUNK_SAMPLES, dtype=np.float32)
    middle = count // 2
    target = big[middle * CHUNK_SAMPLES : (middle + 1) * CHUNK_SAMPLES]
    result = codec.decode(encoded[middle], out=target)
    result_array = np.asarray(result)
    returned_target = result is target
    shares = bool(np.shares_memory(result_array, target))
    filled = bool(np.array_equal(target, expected[middle]))
    others_zero = not bool(np.any(big[: middle * CHUNK_SAMPLES]))
    print(
        f"decode out=slice: result is target {returned_target}, "
        f"shares memory {shares}, slice filled {filled}, "
        f"neighbors untouched {others_zero}"
    )

    tracemalloc.start()
    codec.decode(encoded[middle], out=target)
    _, peak_out = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tracemalloc.start()
    codec.decode(encoded[middle])
    _, peak_plain = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    chunk_bytes = CHUNK_SAMPLES * 4
    print(
        f"decode peak traced allocation: out=slice {peak_out} bytes, "
        f"no out {peak_plain} bytes, chunk {chunk_bytes} bytes"
    )

    def decode_into_out() -> None:
        for index, buf in enumerate(encoded):
            codec.decode(
                buf,
                out=big[index * CHUNK_SAMPLES : (index + 1) * CHUNK_SAMPLES],
            )

    def decode_then_copy() -> None:
        for index, buf in enumerate(encoded):
            big[index * CHUNK_SAMPLES : (index + 1) * CHUNK_SAMPLES] = (
                np.frombuffer(codec.decode(buf), dtype=np.float32)
            )

    out_ms = median_ms(decode_into_out, DECODE_REPEATS)
    copy_ms = median_ms(decode_then_copy, DECODE_REPEATS)
    print(f"decode {count} chunks with out=slice: {out_ms:.2f} ms")
    print(f"decode {count} chunks then copy into slice: {copy_ms:.2f} ms")
    print(f"copy cost over {count} chunks: {copy_ms - out_ms:.2f} ms")


def main() -> None:
    """Write the bundle, then print every timing."""
    print(f"timeseries-zarr {tsz.__version__}")
    print(
        f"bundle: {CHANNEL_COUNT} channels, {SAMPLE_COUNT} samples, write_var"
    )
    data = make_data()
    with tempfile.TemporaryDirectory() as directory:
        output_path = Path(directory) / "bench.zarr"
        begin = time.perf_counter()
        tsz.write_bundle(
            [BenchSource(data)],
            [],
            output_path=output_path,
            opts=tsz.WriteOpts(write_var=True),
        )
        print(f"write_bundle: {(time.perf_counter() - begin) * 1000.0:.0f} ms")
        with tsz.open_bundle(output_path) as bundle:
            bench_stack(bundle)
            bench_channel(bundle)
    bench_decode(data)


if __name__ == "__main__":
    main()
