import pytest
from ..binary_search import binary_search

@pytest.mark.parametrize("arr, target, expected", [
    ([1, 3, 5, 7, 9, 11, 13, 15], 7, 3),
    ([1, 3, 5, 7, 9, 11, 13, 15], 8, -1),
    ([], 5, -1),
    ([5], 5, 0),
    ([5], 7, -1),
    ([2, 4, 4, 6, 8, 8, 10], 4, 1),
])
def test_binary_search(arr, target, expected):
    assert binary_search(arr, target) == expected