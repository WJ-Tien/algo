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


def rotate(nums: list[int], k: int) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """

    k = k % len(nums)
    # reverse whole array
    # reverse first k
    # reverse the remaining

    l, r = 0, len(nums) - 1 # noqa
    while l <= r:
        nums[l], nums[r] = nums[r], nums[l]
        l += 1 # noqa
        r -= 1
    
    l = 0 #noqa
    first_k = l + k - 1
    remaining, r = first_k + 1, len(nums) - 1

    while l <= first_k:
        nums[l], nums[first_k] = nums[first_k], nums[l]
        l += 1 #noqa
        first_k -= 1

    while remaining <= r:
        nums[remaining], nums[r] = nums[r], nums[remaining]
        remaining += 1
        r -= 1
    
def spiralOrder(matrix: list[list[int]]) -> list[int]:

    left, right = 0, len(matrix[0]) - 1
    top, bottom = 0, len(matrix) - 1
    ans = []

    while left <= right and top <= bottom:

        for i in range(left, right + 1):
            ans.append(matrix[top][i])
        top += 1 
        
        for i in range(top, bottom + 1):
            ans.append(matrix[i][right])
        right -= 1 

        if top <= bottom:
            for i in range(right, left - 1, -1):
                ans.append(matrix[bottom][i])
            bottom -= 1
        
        if left <= right:
            for i in range(bottom, top-1, -1):
                ans.append(matrix[i][left])
            left += 1
        
    return ans

def maxProduct(nums: list[int]) -> int:

    max_prod = nums[0]
    min_prod = nums[0] # handle neg * neg = pos cases
    ans = nums[0]

    for i in range(1, len(nums)):
        if nums[i] < 0:
            # key, swap min max when num < 0
            min_prod, max_prod = max_prod, min_prod
        
        # kinda like kadane's algo !
        min_prod = min(nums[i], min_prod * nums[i])
        max_prod = max(nums[i], max_prod * nums[i])

        ans = max(ans, max_prod)
    return ans

def productExceptSelf(nums: list[int]) -> list[int]:
    # prefix product like
    ret = [1] * len(nums)

    # ret[i-1] = prefix_prod before i - 1 (0~i-2)
    # prod[0~i) = prefix_prod[0 ~ i - 2] * nums[i-1]

    for i in range(1, len(nums)):
        ret[i] = ret[i-1] * nums[i - 1]
    
    rprod = 1
    for i in range(len(nums)-1, -1, -1):
        ret[i] *= rprod # the order of this two lines is critical
        rprod *= nums[i]

    return ret