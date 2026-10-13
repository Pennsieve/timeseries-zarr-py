import numpy as np
import pytest

from timeseries_zarr.columns import (
    channels_rows,
    channels_width,
    text_rows,
    text_width,
)


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


def _reference_channels(key_sets, max_key):
    """The reference generator's bitmask encoder."""
    out = np.zeros((len(key_sets), (max_key + 1 + 7) // 8), np.uint8)
    for i, keys in enumerate(key_sets):
        for k in keys:
            out[i, k // 8] |= 1 << (k % 8)
    return out


@pytest.mark.parametrize(
    ("max_key", "expected"),
    [(0, 1), (1, 1), (7, 1), (8, 2), (15, 2), (16, 3), (255, 32), (256, 33)],
)
def test_channels_width_is_one_bit_per_key_rounded_up_to_bytes(
    max_key, expected
):
    assert channels_width(max_key) == expected


def test_channels_width_rejects_a_negative_max_key():
    with pytest.raises(ValueError, match="max_key"):
        channels_width(-1)


def test_channels_rows_equals_the_reference_encoder():
    key_sets = [(0,), (), (1, 3), (0, 5, 9, 17), (4,), tuple(range(18))]
    assert np.array_equal(
        channels_rows(key_sets, 17), _reference_channels(key_sets, 17)
    )


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        ([0], [1, 0]),
        ([7], [128, 0]),
        ([8], [0, 1]),
        ([0, 1, 9], [3, 2]),
        ([15], [0, 128]),
        ([], [0, 0]),
    ],
)
def test_channels_rows_places_key_k_at_bit_k_mod_8_of_byte_k_div_8(
    keys, expected
):
    assert channels_rows([keys], 15).tolist() == [expected]


def test_channels_rows_unpacks_back_to_the_keys_least_significant_bit_first():
    key_sets = [[0, 3, 12], [9], [], [1, 2]]
    rows = channels_rows(key_sets, 12)
    bits = np.unpackbits(rows, axis=1, bitorder="little")
    keys = [np.flatnonzero(row).tolist() for row in bits]
    assert keys == [[0, 3, 12], [9], [], [1, 2]]


def test_channels_rows_empty_list_is_a_row_of_zeros_meaning_the_whole_recording():
    rows = channels_rows([[], [2], []], 7)
    assert rows.tolist() == [[0], [4], [0]]


def test_channels_rows_repeated_and_unordered_keys_set_each_bit_once():
    assert channels_rows([[5, 1, 5, 1, 1]], 7).tolist() == [[34]]


def test_channels_rows_accepts_the_largest_key():
    rows = channels_rows([[16]], 16)
    assert rows.shape == (1, 3)
    assert rows.tolist() == [[0, 0, 1]]


def test_channels_rows_shape_and_dtype():
    rows = channels_rows([[0], [1], []], 20)
    assert rows.shape == (3, 3)
    assert rows.dtype == np.uint8


def test_channels_rows_empty_sequence_yields_zero_rows():
    assert channels_rows([], 9).shape == (0, 2)


def test_channels_rows_accepts_numpy_integer_keys_and_generators():
    rows = channels_rows([np.array([0, 9]), (k for k in (1, 2))], 9)
    assert rows.tolist() == [[1, 2], [6, 0]]


def test_channels_rows_equals_the_reference_on_many_random_key_sets():
    rng = np.random.default_rng(1)
    key_sets = [
        tuple(rng.integers(0, 40, int(rng.integers(0, 6))).tolist())
        for _ in range(3000)
    ]
    assert np.array_equal(
        channels_rows(key_sets, 39), _reference_channels(key_sets, 39)
    )


@pytest.mark.parametrize("keys", [[-1], [0, 8], [5, 100]])
def test_channels_rows_rejects_a_key_outside_zero_to_max_key(keys):
    with pytest.raises(ValueError, match="between 0 and 7"):
        channels_rows([keys], 7)


def test_channels_rows_rejects_a_negative_max_key():
    with pytest.raises(ValueError, match="max_key"):
        channels_rows([[0]], -1)
