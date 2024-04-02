"""
    DESC: swap when arr[j+1] > arr[j]
    ASC: swap when arr[j+1] < arr[j]
"""

def bubble(arr: list) -> None:
    # deal with n - 1 round
    # for this last round, it's not required since we've finished the sorting
    # for each round, the "local max" element of that round is fixed 
    # so we minus i for each round
    # ASC: right-hand side; DESC: left-hand side
    for i in range(len(arr) - 1):
        for j in range(len(arr) - 1 - i):
            if arr[j+1] < arr[j]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

def selection(arr: list) -> None:
    # each round pick the ith element, and loop over the remaining [i+1: n] elements
    # if jth element < ith element, record that idx (swp = j)
    # after the loop, swap them, increase i by 1
    # run n - 1 round 
    i = 0
    while i < len(arr) - 1:
        swp = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[swp]: 
                swp = j
        arr[i], arr[swp] = arr[swp], arr[i]
        i += 1

def insertion(arr: list) -> None:
    # loop over each element and compare that element to all elements 
    # before that element
    # need to loop till the end since the last element might be the smallest
    # kinda like selection, except that insertion keep swaping
    # while selection find min and swap 1 time
    for i in range(len(arr)):
        j = i
        while j - 1 >= 0:
            if arr[j] < arr[j-1]:
                arr[j], arr[j-1] = arr[j-1], arr[j]
            j -= 1

def radix(arr: list) -> None:
    pass

def merge(arr: list) -> None:
    pass

def quick(arr: list) -> None:
    pass


if __name__ == "__main__":
    arr = [3, 2, 7, 1, 4]
    print("Before: ", arr)
    insertion(arr)
    print("After: ", arr)