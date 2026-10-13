"""Per-event column encoders for event channels."""

from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt


def text_width(strings: Sequence[str]) -> int:
    """Return the width in bytes of the longest string once encoded as UTF-8.

    Counts bytes, so a character outside ASCII counts for 2 to 4. An empty
    sequence, or one holding only empty strings, gives 0. Raises ValueError if
    a string cannot be encoded as UTF-8, for example one with an unpaired
    surrogate.
    """
    return max((len(text.encode("utf-8")) for text in strings), default=0)


def text_rows(strings: Sequence[str], width: int) -> npt.NDArray[np.uint8]:
    """Encode strings as rows of UTF-8 bytes, each zero-padded to width.

    Returns a uint8 array of shape (len(strings), width). Row i holds the
    UTF-8 bytes of strings[i] followed by zero bytes, so an empty string is a
    row of zeros. No string may hold a NUL character, since it would read back
    as padding. An empty sequence yields zero rows. Raises ValueError if width
    is negative, a string encodes to more than width bytes, a string holds a
    NUL, or a string cannot be encoded as UTF-8.
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
    second. Raises ValueError if max_key is negative.
    """
    if max_key < 0:
        raise ValueError(f"max_key must be at least 0, got {max_key}")
    return (max_key + 1 + 7) // 8


def channels_rows(
    key_lists: Sequence[Iterable[int]], max_key: int
) -> npt.NDArray[np.uint8]:
    """Encode the channel keys each event applies to as bitmask rows.

    Returns a uint8 array of shape (len(key_lists), channels_width(max_key)).
    Row i has bit k mod 8 of byte k // 8 set, least significant bit first, for
    each key k in key_lists[i]. An empty list gives a row of zeros, which
    means the event applies to the whole recording. A repeated key sets its
    bit once, and keys may come in any order. An empty sequence yields zero
    rows. Raises ValueError if max_key is negative or a key is negative or
    above max_key.
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
