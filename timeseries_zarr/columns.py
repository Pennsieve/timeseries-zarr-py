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


def channels_width(max_key: int) -> int:
    """Return the byte width of a channel bitmask row for keys up to max_key.

    One bit per channel key, so key 0 to 7 need 1 byte and key 8 needs a
    second.

    Args:
        max_key: Largest channel key a row can hold.

    Returns:
        The row width in bytes.

    Raises:
        ValueError: max_key is negative.
    """
    if max_key < 0:
        raise ValueError(f"max_key must be at least 0, got {max_key}")
    return (max_key + 1 + 7) // 8


def channels_rows(
    key_lists: Sequence[Iterable[int]], max_key: int
) -> npt.NDArray[np.uint8]:
    """Encode the channel keys each event applies to as bitmask rows.

    Row i has bit k mod 8 of byte k // 8 set, least significant bit first, for
    each key k in key_lists[i]. An empty list gives a row of zeros, which
    means the event applies to the whole recording. A repeated key sets its
    bit once, and keys may come in any order. An empty sequence yields zero
    rows.

    Args:
        key_lists: The channel keys of each event.
        max_key: Largest channel key allowed.

    Returns:
        A uint8 array of shape (len(key_lists), channels_width(max_key)).

    Raises:
        ValueError: max_key is negative, or a key is negative or above
            max_key.
    """
    width = channels_width(max_key)
    lists = [np.fromiter(keys, dtype=np.int64) for keys in key_lists]
    rows = np.zeros((len(lists), width), dtype=np.uint8)
    if not lists:
        return rows
    keys = np.concatenate(lists)
    if keys.size and (keys.min() < 0 or keys.max() > max_key):
        raise ValueError(
            f"channel keys must lie between 0 and {max_key}, "
            f"got {int(keys.min())} to {int(keys.max())}"
        )
    owner = np.repeat(np.arange(len(lists)), [len(row) for row in lists])
    bits = (1 << (keys & 7)).astype(np.uint8)
    np.bitwise_or.at(rows, (owner, keys >> 3), bits)
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
    """Decode bitmask rows into the channel keys each row names.

    Bit k mod 8 of byte k // 8, least significant bit first, names channel
    key k. Row i gives its keys in ascending order. A row of zeros gives an
    empty list, which means the event applies to the whole recording. An
    array with no rows gives an empty list.

    Args:
        rows: Rank 2 uint8 array of bitmask rows.

    Returns:
        The ascending channel keys of each row.

    Raises:
        ValueError: rows is not a rank 2 uint8 array.
    """
    _check_rows(rows)
    bits = np.unpackbits(rows, axis=1, bitorder="little")
    return [np.flatnonzero(row).tolist() for row in bits]
