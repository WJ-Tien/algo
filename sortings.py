"""
    DESC: swap when arr[j+1] > arr[j]
    ASC: swap when arr[j+1] < arr[j]
"""
from math import log10, floor, pow

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
    pass

def quick(arr: list) -> None:
    pass


if __name__ == "__main__":
    arr = [3, 2, -7, 0,  1, 4]
    print("Before: ", arr)
    arr = radix(arr)
    print("After: ", arr)