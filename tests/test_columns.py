import numpy as np
import pytest

from timeseries_zarr.columns import text_rows, text_width


def _reference_rows(strings, width):
    """The reference generator's loop, with the width given."""
    out = np.zeros((len(strings), width), np.uint8)
    for i, text in enumerate(strings):
        raw = text.encode("utf-8")
        out[i, : len(raw)] = np.frombuffer(raw, np.uint8)
    return out


_MIXED = [
    "seizure onset",
    "",
    "caf\N{LATIN SMALL LETTER E WITH ACUTE}",
    "\N{CJK UNIFIED IDEOGRAPH-65E5}\N{CJK UNIFIED IDEOGRAPH-672C}\N{CJK UNIFIED IDEOGRAPH-8A9E}",
    "\U0001f9e0",
    "x",
]


@pytest.mark.parametrize(
    ("strings", "expected"),
    [
        (["abc", "a", ""], 3),
        (["caf\N{LATIN SMALL LETTER E WITH ACUTE}"], 5),
        (
            [
                "\N{CJK UNIFIED IDEOGRAPH-65E5}\N{CJK UNIFIED IDEOGRAPH-672C}\N{CJK UNIFIED IDEOGRAPH-8A9E}"
            ],
            9,
        ),
        (["\U0001f9e0"], 4),
        (["ab", "\N{LATIN SMALL LETTER E WITH ACUTE}"], 2),
        ([], 0),
        (["", ""], 0),
    ],
)
def test_text_width_counts_utf8_bytes_of_the_longest_string(strings, expected):
    assert text_width(strings) == expected


def test_text_width_rejects_a_string_that_cannot_be_encoded():
    with pytest.raises(ValueError):
        text_width(["ok", "\ud800"])


def test_text_rows_equals_the_reference_loop_on_mixed_text():
    width = text_width(_MIXED)
    assert np.array_equal(
        text_rows(_MIXED, width), _reference_rows(_MIXED, width)
    )


def test_text_rows_has_one_row_per_string_and_uint8_dtype():
    out = text_rows(_MIXED, 20)
    assert out.shape == (len(_MIXED), 20)
    assert out.dtype == np.uint8


def test_text_rows_pads_each_row_with_zero_bytes():
    out = text_rows(["ab", ""], 4)
    assert out.tolist() == [[97, 98, 0, 0], [0, 0, 0, 0]]


def test_text_rows_round_trips_through_utf8():
    out = text_rows(_MIXED, text_width(_MIXED))
    decoded = [bytes(row).rstrip(b"\0").decode("utf-8") for row in out]
    assert decoded == _MIXED


def test_text_rows_accepts_a_string_that_exactly_fills_the_width():
    assert text_rows(["abc"], 3).tolist() == [[97, 98, 99]]


def test_text_rows_empty_sequence_yields_zero_rows():
    assert text_rows([], 5).shape == (0, 5)


def test_text_rows_width_zero_holds_only_empty_strings():
    assert text_rows(["", ""], 0).shape == (2, 0)


def test_text_rows_equals_the_reference_loop_on_many_random_strings():
    rng = np.random.default_rng(0)
    alphabet = list(
        "abcdefg \N{LATIN SMALL LETTER E WITH ACUTE}\N{CJK UNIFIED IDEOGRAPH-65E5}"
    )
    strings = [
        "".join(rng.choice(alphabet, int(rng.integers(0, 12))))
        for _ in range(5000)
    ]
    width = text_width(strings)
    assert np.array_equal(
        text_rows(strings, width), _reference_rows(strings, width)
    )


def test_text_rows_rejects_a_string_wider_than_the_width():
    with pytest.raises(ValueError, match="width"):
        text_rows(["abc", "caf\N{LATIN SMALL LETTER E WITH ACUTE}"], 4)


def test_text_rows_rejects_a_nul_character():
    with pytest.raises(ValueError, match="NUL"):
        text_rows(["a\0b"], 8)


def test_text_rows_rejects_a_negative_width():
    with pytest.raises(ValueError, match="width"):
        text_rows(["a"], -1)


def test_text_rows_rejects_a_string_that_cannot_be_encoded():
    with pytest.raises(ValueError):
        text_rows(["\ud800"], 8)
