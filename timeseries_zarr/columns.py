"""Per-event column encoders for event channels."""

from collections.abc import Sequence

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
