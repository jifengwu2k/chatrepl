# -*- coding: utf-8 -*-
# Copyright (c) 2026 Jifeng Wu
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""Tests for the exact-edit engine used by the edit tool."""

from chatrepl import apply_exact_edits, find_all_occurrences


def test_find_all_occurrences_basic():
    assert find_all_occurrences("banana", "na") == [2, 4]


def test_find_all_occurrences_empty_needle():
    assert find_all_occurrences("banana", "") == []


def test_find_all_occurrences_no_match():
    assert find_all_occurrences("banana", "zz") == []


def test_apply_exact_edits_single_replacement():
    original = "alpha\nbeta\n"
    updated = apply_exact_edits(original, [{"oldText": "beta", "newText": "gamma"}])
    assert updated == "alpha\ngamma\n"


def test_apply_exact_edits_multiple_replacements():
    original = "one two three"
    edits = [
        {"oldText": "one", "newText": "1"},
        {"oldText": "three", "newText": "3"},
    ]
    assert apply_exact_edits(original, edits) == "1 two 3"


def test_apply_exact_edits_adjacent_non_overlapping():
    original = "abcdef"
    edits = [
        {"oldText": "ab", "newText": "AB"},
        {"oldText": "cd", "newText": "CD"},
    ]
    assert apply_exact_edits(original, edits) == "ABCDef"


def test_apply_exact_edits_unicode():
    original = u"\u4f60\u597d world"
    updated = apply_exact_edits(
        original,
        [{"oldText": u"\u4f60\u597d", "newText": u"\u4e16\u754c"}],
    )
    assert updated == u"\u4e16\u754c world"


def test_apply_exact_edits_edits_out_of_order_applied_in_position_order():
    original = "aaa bbb"
    edits = [
        {"oldText": "bbb", "newText": "2"},
        {"oldText": "aaa", "newText": "1"},
    ]
    assert apply_exact_edits(original, edits) == "1 2"


def test_apply_exact_edits_no_change():
    assert apply_exact_edits("abc", [{"oldText": "abc", "newText": "abc"}]) == "abc"


def test_apply_exact_edits_not_found_raises_with_index():
    try:
        apply_exact_edits("abc", [{"oldText": "zzz", "newText": "x"}])
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Edit 0" in str(exc)
        assert "not found" in str(exc)


def test_apply_exact_edits_duplicate_match_raises():
    try:
        apply_exact_edits("xx xx", [{"oldText": "xx", "newText": "y"}])
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "must match exactly once" in str(exc)


def test_apply_exact_edits_empty_old_text_raises():
    try:
        apply_exact_edits("abc", [{"oldText": "", "newText": "y"}])
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "oldText must not be empty" in str(exc)


def test_apply_exact_edits_overlapping_raises():
    edits = [
        {"oldText": "abc", "newText": "X"},
        {"oldText": "bc", "newText": "Y"},
    ]
    try:
        apply_exact_edits("abcdef", edits)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "overlap" in str(exc)


def test_apply_exact_edits_second_edit_matched_against_original():
    # Each edit's oldText must match exactly once in the ORIGINAL file, not in
    # the partially-updated text: an oldText that only appears as the result of
    # an earlier replacement is still rejected as not found.
    original = "ab"
    edits = [
        {"oldText": "ab", "newText": "abc"},
        {"oldText": "abc", "newText": "d"},
    ]
    try:
        apply_exact_edits(original, edits)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Edit 1" in str(exc)
        assert "not found" in str(exc)
