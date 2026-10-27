# Changelog

Notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

Writes TS-Zarr 1.0 bundles from continuous and event channel sources.

Writes every channel of a `MultiChannelSource` in one pass over its windows.

Writes each continuous channel in one pass. The writer folds every level while the
samples are in memory, encodes each shard as its chunks complete, and reads nothing back
from the store. The memory of a channel no longer grows with the shard size, and
`write_var` no longer changes the memory or the algorithm. The stored `minmax` and `count`
match the earlier writer. `mean` matches it without `write_var` to float32 rounding.

Reads TS-Zarr 1.0 bundles, with extras for pandas, xarray, and remote stores. Requires
zarr 3.2.1 or later.

`ContinuousChannel.read` takes `members`, the level members to read. A read with
`members=()` holds `min` and `max` alone.

`WriteOpts.max_workers` sets the threads that fold and encode the channels of a window. It
defaults to the CPUs available to the process. The writer reads the next window while the threads write the
previous one, so a source must not overwrite the arrays of a window it has yielded. A
bundle is byte-identical at any thread count.

A window of a `MultiChannelSource` may be a `(rows, channels)` array. The writer
transposes it once into contiguous rows for each channel.

`Bundle.stack` and `stack_channels` read one array per channel and take
`max_workers`, the number of threads that read the channels.
