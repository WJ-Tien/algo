"""
    DESC: swap when arr[j+1] > arr[j]
    ASC: swap when arr[j+1] < arr[j]
"""
import random #noqa
from math import log10, floor, pow

def bubble(arr: list) -> None:
    # TC: O(n^2)
    # SC: O(1)
    # deal with n - 1 round
    # for this last round, it's not required since we've finished the sorting
    # for each round, the "local max" element of that round is fixed 
    # so we minus i for each round
    # ASC: right-hand side; DESC: left-hand side
    for i in range(len(arr) - 1):
        for j in range(len(arr) - 1 - i):
            if arr[j+1] < arr[j]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

def bubble_sort_optimized(arr: list) -> None:
    n = len(arr)
    for i in range(n - 1):
        swapped = False  # 記錄本輪是否發生交換
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:  # 兩兩比較
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True  # 如果發生交換，標記為 True
        if not swapped:
            break  # 如果沒有交換，代表已經排序完成，提前結束

def selection(arr: list) -> None:
    # TC: O(n^2)
    # SC: O(1)
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
    # TC: O(n^2)
    # SC: O(1)
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

def radix(arr: list) -> list:
    # TC: O(nd). d = max_digit
    # SC: O(n + k). k = 10 here

    # The idea behind radix sort is that num with longer num of digit
    # is larger :). The early the num goes to lower bucket_idx --> smaller
    # kinda like stack LIFO ops
    # 1. get max num of digit in the arr to determine n_iters
    # 2. create a bucket to store temporary result (k = 10)
    # 3. loop over n_iters times
    # 4. assign the num to bucket_idx by get_digit
    # 5. concat all sub array to a array 
    # 6. repeat 3 ~ 5 
    # handle postive and negative ints separately

    def get_digit(num: int, idx: int) -> int:
        return floor(abs(num)//(pow(10, idx))) % 10

    def get_digit_count(num: int) -> int:
        if num == 0:
            return 1
        return floor(log10(abs(num))) + 1

    def max_n_digits(arr: list) -> int:
        max_digit = 0
        for num in arr:
            max_digit = max(max_digit, get_digit_count(num))
        return max_digit

    positive_list = [num for num in arr if num >= 0]
    negative_list = [num for num in arr if num < 0]

    positive_n_iters = max_n_digits(positive_list) 
    negative_n_iters = max_n_digits(negative_list) 

    for n in range(positive_n_iters):
        bucket = [[] for _ in range(10)] # k = 10
        for num in positive_list:
            bucket_idx = get_digit(num, n)
            bucket[bucket_idx].append(num)

        positive_list = [element for buc in bucket for element in buc] # n

    for n in range(negative_n_iters):
        bucket = [[] for _ in range(10)] # k = 10
        for num in negative_list:
            bucket_idx = get_digit(num, n)
            bucket[bucket_idx].append(num)

        negative_list = [element for buc in bucket for element in buc] # n

    negative_list.reverse()
    negative_list.extend(positive_list)

    return negative_list

def merge(arr: list) -> None:
    # TC: O(nlogn)
    # SC: O(n)
    # split till the array contains only 1 element
    # sort it (this does merge as well)
    # do above steps recursively
    
    if len(arr) <= 1:
        return

    # split
    left_arr = arr[:len(arr)//2]
    right_arr = arr[len(arr)//2:]
    merge(left_arr)
    merge(right_arr)

    # merge 
    # with order implicitly 
    i = j = k = 0
    while i < len(left_arr) and j < len(right_arr):
        if left_arr[i] < right_arr[j]:
            arr[k] = left_arr[i] 
            i += 1
        else:
            arr[k] = right_arr[j]
            j += 1
        k += 1
    
    while i < len(left_arr):
        arr[k] = left_arr[i]
        k += 1
        i += 1

    while j < len(right_arr):
        arr[k] = right_arr[j]
        k += 1
        j += 1

def _partition(arr: list, left: int, right: int) -> int:
    # return end index
    # if random pivot
    # pivot_idx = random.randint(left, right)
    # arr[pivot_idx], arr[right] = arr[right], arr[pivot_idx]

    i = left
    j = right - 1
    pivot = arr[right]
    while i < j:
        while i < right and arr[i] < pivot:
            i += 1
        while j > left and arr[j] >= pivot:
            j -= 1
        if i < j:
            arr[i], arr[j] = arr[j], arr[i]
    if arr[i] > pivot:
        arr[i], arr[right] = arr[right], arr[i]
    return i

def quick(arr: list, left: int, right: int) -> None:
    # TC: O(nlogn)
    # SC: O(logn)
    if left >= right:
        return
    partition_pos = _partition(arr, left, right) 
    quick(arr, left, partition_pos - 1)
    quick(arr, partition_pos + 1, right)
