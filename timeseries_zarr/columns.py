"""Per-event column encoders and decoders for event channels."""

from collections.abc import Callable, Iterable, Sequence

import numpy as np
import numpy.typing as npt

_ROW_RANK = 2


def text_width(strings: Sequence[str]) -> int:
    """Return the width in bytes of the longest string once encoded as UTF-8.

    Counts bytes, so a character outside ASCII counts for 2 to 4. An empty
    sequence, or one holding only empty strings, gives 0.

    Args:
        strings: Strings to measure.

    Returns:
        The width in bytes.

    Raises:
        ValueError: A string cannot be encoded as UTF-8, for example one with
            an unpaired surrogate.
    """
    return max((len(text.encode("utf-8")) for text in strings), default=0)


def text_rows(strings: Sequence[str], width: int) -> npt.NDArray[np.uint8]:
    """Encode strings as rows of UTF-8 bytes, each zero-padded to width.

    Row i holds the UTF-8 bytes of strings[i] followed by zero bytes, so an
    empty string is a row of zeros. An empty sequence yields zero rows.

    Args:
        strings: Strings to encode. None may hold a NUL character, which
            would read back as padding.
        width: Row width in bytes.

    Returns:
        A uint8 array of shape (len(strings), width).

    Raises:
        ValueError: The width is negative, a string encodes to more than width
            bytes, a string holds a NUL, or a string cannot be encoded as
            UTF-8.
    """
    if width < 0:
        raise ValueError(f"width must be at least 0, got {width}")
    encoded = [text.encode("utf-8") for text in strings]
    for index, raw in enumerate(encoded):
        if len(raw) > width:
            raise ValueError(
                f"string {index} takes {len(raw)} bytes, over the width {width}"
            )
        if b"\0" in raw:
            raise ValueError(f"string {index} holds a NUL character")
    lengths = np.array([len(raw) for raw in encoded], dtype=np.int64)
    rows = np.zeros((len(encoded), width), dtype=np.uint8)
    inside = np.arange(width)[None, :] < lengths[:, None]
    rows[inside] = np.frombuffer(b"".join(encoded), dtype=np.uint8)
    return rows


def channels_width(max_index: int) -> int:
    """Return the byte width of a channel bitmask row for indices up to max_index.

    One bit per channel index, so index 0 to 7 need 1 byte and index 8 needs a
    second.

    Args:
        max_index: Largest channel index a row can hold.

    Returns:
        The row width in bytes.

    Raises:
        ValueError: max_index is negative.
    """
    if max_index < 0:
        raise ValueError(f"max_index must be at least 0, got {max_index}")
    return (max_index + 1 + 7) // 8


def channels_rows(
    index_lists: Sequence[Iterable[int]], max_index: int
) -> npt.NDArray[np.uint8]:
    """Encode the channel indices each event applies to as bitmask rows.

    Row i has bit k mod 8 of byte k // 8 set, least significant bit first, for
    each index k in index_lists[i]. An empty list gives a row of zeros, which
    names no channel. A repeated index sets its bit once, and indices may come in
    any order. An empty sequence yields zero
    rows.

    Args:
        index_lists: The channel indices of each event.
        max_index: Largest channel index allowed.

    Returns:
        A uint8 array of shape (len(index_lists), channels_width(max_index)).

    Raises:
        ValueError: max_index is negative, or an index is negative or above
            max_index.
    """
    width = channels_width(max_index)
    lists = [np.fromiter(indices, dtype=np.int64) for indices in index_lists]
    rows = np.zeros((len(lists), width), dtype=np.uint8)
    if not lists:
        return rows
    indices = np.concatenate(lists)
    if indices.size and (indices.min() < 0 or indices.max() > max_index):
        raise ValueError(
            f"channel indices must lie between 0 and {max_index}, "
            f"got {int(indices.min())} to {int(indices.max())}"
        )
    owner = np.repeat(np.arange(len(lists)), [len(row) for row in lists])
    bits = (1 << (indices & 7)).astype(np.uint8)
    np.bitwise_or.at(rows, (owner, indices >> 3), bits)
    return rows


def _check_rows(rows: npt.NDArray[np.uint8]) -> None:
    """Raise ValueError unless rows is a rank 2 uint8 array."""
    if rows.ndim != _ROW_RANK:
        raise ValueError(f"rows must have rank 2, got shape {rows.shape}")
    if rows.dtype != np.uint8:
        raise ValueError(f"rows must have dtype uint8, got {rows.dtype}")


def decode_text_rows(rows: npt.NDArray[np.uint8]) -> list[str]:
    """Decode rows of zero-padded UTF-8 bytes into one string per row.

    Row i decodes the bytes before its first zero byte, or the whole row when
    it holds no zero byte. A row of zeros gives the empty string, and so does
    every row of width 0. An array with no rows gives an empty list.

    Args:
        rows: Rank 2 uint8 array of zero-padded UTF-8 rows.

    Returns:
        One string per row.

    Raises:
        ValueError: rows is not a rank 2 uint8 array or a row is not valid
            UTF-8.
    """
    _check_rows(rows)
    return _decoded(
        rows, range(rows.shape[0]), lambda index: f"text row {index}"
    )


def decode_event_text(
    rows: npt.NDArray[np.uint8], indices: Iterable[int], where: str
) -> list[str]:
    """Decode the text rows of events as decode_text_rows does.

    Args:
        rows: Rank 2 uint8 array of zero-padded UTF-8 rows.
        indices: The event index of each row, one per row, for error messages.
        where: Channel name for error messages.

    Returns:
        One string per row.

    Raises:
        ValueError: rows is not a rank 2 uint8 array or a row is not valid
            UTF-8.
    """
    _check_rows(rows)
    return _decoded(rows, indices, lambda index: f"{where}: event {index} text")


def _decoded(
    rows: npt.NDArray[np.uint8],
    indices: Iterable[int],
    subject: Callable[[int], str],
) -> list[str]:
    """Decode each row up to its first zero byte as UTF-8.

    Args:
        rows: Rank 2 uint8 array of zero-padded UTF-8 rows.
        indices: The number of each row, one per row.
        subject: Names a row by its number in an error message.

    Returns:
        One string per row.

    Raises:
        ValueError: A row is not valid UTF-8.
    """
    strings = []
    for index, row in zip(indices, rows, strict=True):
        text = row.tobytes().partition(b"\0")[0]
        try:
            strings.append(text.decode("utf-8"))
        except UnicodeDecodeError as error:
            raise ValueError(
                f"{subject(index)} is not valid UTF-8 at byte {error.start}"
            ) from error
    return strings


def decode_channels_rows(rows: npt.NDArray[np.uint8]) -> list[list[int]]:
    """Decode bitmask rows into the channel indices each row names.

    Bit k mod 8 of byte k // 8, least significant bit first, names channel
    index k. Row i gives its indices in ascending order. A row of zeros gives an
    empty list, which names no channel. An array with no rows gives an empty
    list.

    Args:
        rows: Rank 2 uint8 array of bitmask rows.

    Returns:
        The ascending channel indices of each row.

    Raises:
        ValueError: rows is not a rank 2 uint8 array.
    """
    _check_rows(rows)
    bits = np.unpackbits(rows, axis=1, bitorder="little")
    return [np.flatnonzero(row).tolist() for row in bits]


def check_channels_rows(
    rows: npt.NDArray[np.uint8],
    channel_count: int,
    indices: Iterable[int],
    where: str,
) -> None:
    """Raise ValueError when a bitmask row sets a bit for a missing channel.

    Bits at a position of channel_count or above name no channel of the bundle.

    Args:
        rows: Rank 2 uint8 array of bitmask rows.
        channel_count: The number of channels in the bundle.
        indices: The event index of each row, one per row, for error messages.
        where: Channel name for error messages.

    Raises:
        ValueError: rows is not a rank 2 uint8 array, or a row sets a bit at
            position channel_count or above.
    """
    _check_rows(rows)
    bits = np.unpackbits(rows, axis=1, bitorder="little")[:, channel_count:]
    if not bits.any():
        return
    row, column = np.argwhere(bits)[0]
    index = list(indices)[int(row)]
    raise ValueError(
        f"{where}: event {index} sets channel bit {channel_count + int(column)}"
        f", and the bundle has {channel_count} channels"
    )
