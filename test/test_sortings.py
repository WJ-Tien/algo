import pytest
from ..sortings import bubble, selection, insertion, radix, merge, quick

@pytest.mark.parametrize("arr, expected", [
    ([5, 2, 4, 6, 1, 3], [1, 2, 3, 4, 5, 6]),
    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
    ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
    ([], []),
    ([1], [1]),
    ([1, 1, 1, 1], [1, 1, 1, 1]),
    ([5, -2, 4, -6, 1, 3], [-6, -2, 1, 3, 4, 5]),
])
def test_bubble_sort(arr, expected):
    bubble(arr)
    assert arr == expected

@pytest.mark.parametrize("arr, expected", [
    ([5, 2, 4, 6, 1, 3], [1, 2, 3, 4, 5, 6]),
    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
    ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
    ([], []),
    ([1], [1]),
    ([1, 1, 1, 1], [1, 1, 1, 1]),
    ([5, -2, 4, -6, 1, 3], [-6, -2, 1, 3, 4, 5]),
])
def test_selection_sort(arr, expected):
    selection(arr)
    assert arr == expected

@pytest.mark.parametrize("arr, expected", [
    ([5, 2, 4, 6, 1, 3], [1, 2, 3, 4, 5, 6]),
    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
    ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
    ([], []),
    ([1], [1]),
    ([1, 1, 1, 1], [1, 1, 1, 1]),
    ([5, -2, 4, -6, 1, 3], [-6, -2, 1, 3, 4, 5]),
])
def test_insertion_sort(arr, expected):
    insertion(arr)
    assert arr == expected

@pytest.mark.parametrize("arr, expected", [
    ([170, 45, 75, 90, 802, 24, 2, 66], [2, 24, 45, 66, 75, 90, 170, 802]),
    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
    ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
    ([], []),
    ([1], [1]),
    ([1, 1, 1, 1], [1, 1, 1, 1]),
    ([50, -20, 40, -60, 10, 30], [-60, -20, 10, 30, 40, 50]),
])
def test_radix_sort(arr, expected):
    assert radix(arr) == expected

@pytest.mark.parametrize("arr, expected", [
    ([5, 2, 4, 6, 1, 3], [1, 2, 3, 4, 5, 6]),
    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
    ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
    ([], []),
    ([1], [1]),
    ([1, 1, 1, 1], [1, 1, 1, 1]),
    ([5, -2, 4, -6, 1, 3], [-6, -2, 1, 3, 4, 5]),
])
def test_merge_sort(arr, expected):
    merge(arr)
    assert arr == expected

@pytest.mark.parametrize("arr, expected", [
    ([5, 2, 4, 6, 1, 3], [1, 2, 3, 4, 5, 6]),
    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
    ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
    ([], []),
    ([1], [1]),
    ([1, 1, 1, 1], [1, 1, 1, 1]),
    ([5, -2, 4, -6, 1, 3], [-6, -2, 1, 3, 4, 5]),
])
def test_quick_sort(arr, expected):
    quick(arr, 0, len(arr) - 1)
    assert arr == expected