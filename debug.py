from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        slow = fast = head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        cur_head = slow
        prev = None

        while cur_head:
            next_node = cur_head.next
            cur_head.next = prev
            prev = cur_head
            cur_head = next_node
        reversed_head = prev
        # head --> ... --> null <-- slow ... <-- reversed_head 
        ori_head = head
        rev_head = reversed_head
        while rev_head.next: # very tricky !
            ori_next_node = ori_head.next # 2
            rev_next_node = rev_head.next # 3
        
            ori_head.next = rev_head # 4
            rev_head.next = ori_next_node # 2
        
            ori_head = ori_next_node # 2
            rev_head = rev_next_node # 3

def create_linked_list():
    # 建立節點
    node1 = ListNode(1)
    node2 = ListNode(2)
    node3 = ListNode(3)
    node4 = ListNode(4)
    
    # 連接節點
    node1.next = node2
    node2.next = node3
    node3.next = node4
    node4.next = None  # 最後一個節點指向 None
    
    return node1  # 返回頭節點

def print_linked_list(head):
    current = head
    while current:
        print(current.val, end="")
        if current.next:
            print(" -> ", end="")
        current = current.next
    print(" -> None")

# head = create_linked_list()
# s = Solution()
# s.reorderList(head)

# a = {"A":1, "B":2}
# for i in a:
# 	print(i)
import time # noqa
from functools import lru_cache # noqa

memo = dict()
def fib_good(n):
    if n in memo:
        return memo[n]
    if n == 0 or n == 1:
        return n

    result = fib_good(n-1) + fib_good(n-2)
    memo[n] = result
    return result

def fib_bad(n):
    if n == 0 or n == 1:
        return n
    return fib_bad(n-1) + fib_bad(n-2)


@lru_cache(maxsize=None)
def fib_best(n):
    if n == 0 or n == 1:
        return n
    return fib_good(n - 1) + fib_good(n - 2)


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

arr = [5,4,3,1,2]

merge(arr)