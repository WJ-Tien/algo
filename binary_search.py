def binary_search(arr: list, target: int) -> int:
    # TC: O(logn)
    # SC: O(1)

    low, high = 0, len(arr) - 1
    # low must == high; othersie if there remains only two figs and target is the higher one
    # the code won't work
    while low <= high:
        mid = low + (high - low) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] > target:
            high = mid - 1
        else:
            low = mid + 1
    return -1