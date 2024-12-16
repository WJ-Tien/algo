from typing import Optional

def sortColors(nums: list[int]) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    # Dutch National Flag Problem DNF
    low, mid, high = 0, 0, len(nums) - 1
    
    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:  # nums[mid] == 2
            # since you cannot guarantee nums[mid]'s value at the moment
            # so you have to check it again, hence no mid += 1
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1

# quick select algorithm
def findKthLargest(nums: list[int], k: int) -> int:
    k_smallest = len(nums) - k

    def quick_select(l, r): # noqa
        p, pivot = l, nums[r] 
        for i in range(l, r):
            if nums[i] <= pivot:
                nums[p], nums[i] = nums[i], nums[p]
                p += 1
        nums[p], nums[r] = nums[r], nums[p]

        if p < k_smallest:
            return quick_select(p+1, r)
        if p > k_smallest:
            return quick_select(l, p - 1)
        return nums[p]
    return quick_select(0, len(nums)-1)

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def swapPairs(head: Optional[ListNode]) -> Optional[ListNode]:
    
    dummy = ListNode()
    cur = dummy.next = head
    prev = dummy

    while cur and cur.next:
        first, second = cur, cur.next
        third_node = cur.next.next
        second.next = first
        prev.next = second # important !
        first.next = third_node
        prev = first
        cur = third_node
        
    return dummy.next 