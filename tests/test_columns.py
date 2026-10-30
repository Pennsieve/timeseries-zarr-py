import re

import numpy as np
import pytest

from timeseries_zarr.columns import (
    channels_rows,
    channels_width,
    check_channels_rows,
    decode_channels_rows,
    decode_text_rows,
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
    "\N{BRAIN}",
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
        (["\N{BRAIN}"], 4),
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


def _reference_channels(index_sets, max_index):
    """The reference generator's bitmask encoder."""
    out = np.zeros((len(index_sets), (max_index + 1 + 7) // 8), np.uint8)
    for i, indices in enumerate(index_sets):
        for k in indices:
            out[i, k // 8] |= 1 << (k % 8)
    return out


@pytest.mark.parametrize(
    ("max_index", "expected"),
    [(0, 1), (1, 1), (7, 1), (8, 2), (15, 2), (16, 3), (255, 32), (256, 33)],
)
def test_channels_width_is_one_bit_per_index_rounded_up_to_bytes(
    max_index, expected
):
    assert channels_width(max_index) == expected


def test_channels_width_rejects_a_negative_max_index():
    with pytest.raises(ValueError, match="max_index"):
        channels_width(-1)


def test_channels_rows_equals_the_reference_encoder():
    index_sets = [(0,), (), (1, 3), (0, 5, 9, 17), (4,), tuple(range(18))]
    assert np.array_equal(
        channels_rows(index_sets, 17), _reference_channels(index_sets, 17)
    )


@pytest.mark.parametrize(
    ("indices", "expected"),
    [
        ([0], [1, 0]),
        ([7], [128, 0]),
        ([8], [0, 1]),
        ([0, 1, 9], [3, 2]),
        ([15], [0, 128]),
        ([], [0, 0]),
    ],
)
def test_channels_rows_places_index_k_at_bit_k_mod_8_of_byte_k_div_8(
    indices, expected
):
    assert channels_rows([indices], 15).tolist() == [expected]


def test_channels_rows_unpacks_back_to_the_indices_least_significant_bit_first():
    index_sets = [[0, 3, 12], [9], [], [1, 2]]
    rows = channels_rows(index_sets, 12)
    bits = np.unpackbits(rows, axis=1, bitorder="little")
    indices = [np.flatnonzero(row).tolist() for row in bits]
    assert indices == [[0, 3, 12], [9], [], [1, 2]]


def test_channels_rows_empty_list_is_a_row_of_zeros_naming_no_channel():
    rows = channels_rows([[], [2], []], 7)
    assert rows.tolist() == [[0], [4], [0]]


def test_channels_rows_repeated_and_unordered_indices_set_each_bit_once():
    assert channels_rows([[5, 1, 5, 1, 1]], 7).tolist() == [[34]]


def test_channels_rows_accepts_the_largest_index():
    rows = channels_rows([[16]], 16)
    assert rows.shape == (1, 3)
    assert rows.tolist() == [[0, 0, 1]]


def test_channels_rows_shape_and_dtype():
    rows = channels_rows([[0], [1], []], 20)
    assert rows.shape == (3, 3)
    assert rows.dtype == np.uint8


def test_channels_rows_empty_sequence_yields_zero_rows():
    assert channels_rows([], 9).shape == (0, 2)


def test_channels_rows_accepts_numpy_integer_indices_and_generators():
    rows = channels_rows([np.array([0, 9]), (k for k in (1, 2))], 9)
    assert rows.tolist() == [[1, 2], [6, 0]]


def test_channels_rows_equals_the_reference_on_many_random_index_sets():
    rng = np.random.default_rng(1)
    index_sets = [
        tuple(rng.integers(0, 40, int(rng.integers(0, 6))).tolist())
        for _ in range(3000)
    ]
    assert np.array_equal(
        channels_rows(index_sets, 39), _reference_channels(index_sets, 39)
    )


@pytest.mark.parametrize("indices", [[-1], [0, 8], [5, 100]])
def test_channels_rows_rejects_an_index_outside_zero_to_max_index(indices):
    with pytest.raises(ValueError, match="between 0 and 7"):
        channels_rows([indices], 7)


def test_channels_rows_rejects_a_negative_max_index():
    with pytest.raises(ValueError, match="max_index"):
        channels_rows([[0]], -1)


def _byte_rows(*rows):
    return np.array(rows, dtype=np.uint8).reshape(len(rows), -1)


def test_decode_text_rows_inverts_text_rows_on_mixed_text():
    rows = text_rows(_MIXED, text_width(_MIXED))
    assert decode_text_rows(rows) == _MIXED


def test_decode_text_rows_inverts_text_rows_on_many_random_strings():
    rng = np.random.default_rng(2)
    alphabet = list(
        "abc {}:\N{LATIN SMALL LETTER E WITH ACUTE}"
        "\N{GREEK SMALL LETTER MU}\N{CJK UNIFIED IDEOGRAPH-65E5}\N{BRAIN}"
    )
    strings = [
        "".join(rng.choice(alphabet, int(rng.integers(0, 16))))
        for _ in range(3000)
    ]
    rows = text_rows(strings, text_width(strings) + 3)
    assert decode_text_rows(rows) == strings


def test_decode_text_rows_reads_a_row_with_no_zero_byte_as_the_whole_row():
    text = "spike-wave discharge, left temporal"
    rows = text_rows([text, "x"], len(text))
    assert np.all(rows[0] != 0)
    assert decode_text_rows(rows) == [text, "x"]


def test_decode_text_rows_reads_a_full_width_row_of_multibyte_characters():
    text = "caf\N{LATIN SMALL LETTER E WITH ACUTE}"
    assert decode_text_rows(text_rows([text], 5)) == [text]


def test_decode_text_rows_stops_at_the_first_zero_byte_inside_a_row():
    rows = _byte_rows([97, 98, 0, 99, 100], [0, 101, 0, 0, 0])
    assert decode_text_rows(rows) == ["ab", ""]


def test_decode_text_rows_ignores_invalid_utf8_after_the_first_zero_byte():
    rows = _byte_rows([97, 0, 0xFF, 0xFE])
    assert decode_text_rows(rows) == ["a"]


def test_decode_text_rows_reads_a_row_of_zeros_as_the_empty_string():
    rows = np.zeros((3, 4), dtype=np.uint8)
    assert decode_text_rows(rows) == ["", "", ""]


def test_decode_text_rows_gives_one_empty_string_per_row_at_width_zero():
    rows = np.zeros((4, 0), dtype=np.uint8)
    assert decode_text_rows(rows) == ["", "", "", ""]


@pytest.mark.parametrize("width", [0, 6])
def test_decode_text_rows_gives_an_empty_list_for_zero_rows(width):
    assert decode_text_rows(np.zeros((0, width), dtype=np.uint8)) == []


def test_decode_text_rows_reads_rows_of_a_strided_view():
    rows = text_rows(["ab", "cd", "ef"], 2)
    assert decode_text_rows(rows[::2]) == ["ab", "ef"]
    assert decode_text_rows(rows.T) == ["ace", "bdf"]


@pytest.mark.parametrize(
    "bad",
    [
        [0xFF, 0, 0],
        [97, 0xC3, 0],
        [0xE6, 0x97],
        [0xED, 0xA0, 0x80],
    ],
)
def test_decode_text_rows_rejects_invalid_utf8_naming_the_row(bad):
    rows = np.zeros((3, len(bad)), dtype=np.uint8)
    rows[0, 0] = 97
    rows[2] = bad
    with pytest.raises(ValueError, match="text row 2 is not valid UTF-8"):
        decode_text_rows(rows)


def test_decode_text_rows_names_the_byte_where_the_utf8_breaks():
    rows = _byte_rows([97, 98, 0xFF, 0])
    with pytest.raises(ValueError, match="row 0 is not valid UTF-8 at byte 2"):
        decode_text_rows(rows)


@pytest.mark.parametrize("shape", [(4,), (2, 3, 4), ()])
def test_decode_text_rows_rejects_an_array_that_is_not_rank_2(shape):
    rows = np.zeros(shape, dtype=np.uint8)
    with pytest.raises(
        ValueError, match=re.escape(f"rank 2, got shape {shape}")
    ):
        decode_text_rows(rows)


@pytest.mark.parametrize("dtype", [np.int64, np.uint16, np.int8])
def test_decode_text_rows_rejects_a_dtype_other_than_uint8(dtype):
    rows = np.zeros((2, 3), dtype=dtype)
    with pytest.raises(ValueError, match="dtype uint8"):
        decode_text_rows(rows)


def test_decode_channels_rows_inverts_channels_rows_on_many_random_index_sets():
    rng = np.random.default_rng(3)
    index_sets = [
        sorted(set(rng.integers(0, 40, int(rng.integers(0, 8))).tolist()))
        for _ in range(3000)
    ]
    assert decode_channels_rows(channels_rows(index_sets, 39)) == index_sets


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ([1, 0], [0]),
        ([128, 0], [7]),
        ([0, 1], [8]),
        ([0b10000001, 0b00000010], [0, 7, 9]),
        ([0, 128], [15]),
        ([255, 255], list(range(16))),
    ],
)
def test_decode_channels_rows_reads_index_k_from_bit_k_mod_8_of_byte_k_div_8(
    row, expected
):
    assert decode_channels_rows(_byte_rows(row)) == [expected]


def test_decode_channels_rows_reads_indices_above_7_across_3_bytes():
    rows = channels_rows([[3, 8, 23], [16, 2]], 23)
    assert rows.shape == (2, 3)
    assert decode_channels_rows(rows) == [[3, 8, 23], [2, 16]]


def test_decode_channels_rows_gives_each_repeated_index_once_in_ascending_order():
    rows = channels_rows([[9, 5, 1, 9, 5, 1]], 15)
    assert decode_channels_rows(rows) == [[1, 5, 9]]


def test_decode_channels_rows_reads_a_row_of_zeros_as_no_channel():
    rows = channels_rows([[], [2], []], 7)
    assert decode_channels_rows(rows) == [[], [2], []]


def test_decode_channels_rows_gives_an_empty_list_for_zero_rows():
    assert decode_channels_rows(np.zeros((0, 2), dtype=np.uint8)) == []


def test_decode_channels_rows_gives_empty_index_lists_at_width_zero():
    rows = np.zeros((2, 0), dtype=np.uint8)
    assert decode_channels_rows(rows) == [[], []]


def test_decode_channels_rows_returns_python_integers():
    indices = decode_channels_rows(channels_rows([[0, 9]], 9))[0]
    assert all(type(index) is int for index in indices)


@pytest.mark.parametrize("shape", [(4,), (2, 3, 4), ()])
def test_decode_channels_rows_rejects_an_array_that_is_not_rank_2(shape):
    rows = np.zeros(shape, dtype=np.uint8)
    with pytest.raises(
        ValueError, match=re.escape(f"rank 2, got shape {shape}")
    ):
        decode_channels_rows(rows)


@pytest.mark.parametrize("dtype", [np.int64, np.bool_])
def test_decode_channels_rows_rejects_a_dtype_other_than_uint8(dtype):
    rows = np.zeros((2, 1), dtype=dtype)
    with pytest.raises(ValueError, match="dtype uint8"):
        decode_channels_rows(rows)


def test_check_channels_rows_accepts_bits_below_the_channel_count():
    rows = channels_rows([[0, 8], [], [3]], 8)
    check_channels_rows(rows, 9, range(3), "channel 2")


def test_check_channels_rows_accepts_a_row_with_no_bit_set():
    check_channels_rows(np.zeros((2, 1), np.uint8), 3, range(2), "channel 2")


def test_check_channels_rows_rejects_a_bit_at_the_channel_count():
    rows = channels_rows([[0], [1, 3]], 7)
    with pytest.raises(
        ValueError,
        match="channel 2: event 11 sets channel bit 3, and the bundle has 3 "
        "channels",
    ):
        check_channels_rows(rows, 3, [10, 11], "channel 2")


def test_check_channels_rows_rejects_a_bit_in_the_last_byte_past_the_count():
    rows = np.array([[0, 0b10000000]], np.uint8)
    with pytest.raises(ValueError, match="sets channel bit 15"):
        check_channels_rows(rows, 12, [0], "channel 2")


def test_check_channels_rows_rejects_rows_that_are_not_uint8_rank_2():
    with pytest.raises(ValueError, match="rank 2"):
        check_channels_rows(np.zeros(3, np.uint8), 3, [0], "channel 2")
