from collections import deque
from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def longestOnes(nums: list[int], k: int) -> int:
    # 1004. Max Consecutive Ones III
    left = 0
    ans = 0
    cur_sum = 0

    for right in range(len(nums)):
        if nums[right] == 0:
            cur_sum += 1
        while cur_sum > k and left <= right:
            # >= k+1's 0 --> reset to <= k's 0
            if nums[left] == 0:
                cur_sum -= 1
            left += 1
        ans = max(ans, right - left + 1)
    return ans

def numSubarrayProductLessThanK(nums: list[int], k: int) -> int:

    # sliding window classic pattern

    if k <= 0:
        return 0

    left = 0
    ans = 0
    prod = 1
    for right in range(len(nums)):
        prod *= nums[right]
        while prod >= k and left <= right: 
            prod //= nums[left]
            left += 1
        ans += (right - left + 1) # key here !
        # window size n == n combinations
    return ans

def find_best_subarray(nums, k):
    curr = 0
    for i in range(k):
        curr += nums[i]
    
    ans = curr
    for i in range(k, len(nums)):
        curr += nums[i] - nums[i - k] 
        # 1 2 3 4 5, k = 4
        # s = 1+2+3+4 --> s + 5 - 1 --> sliding window
        ans = max(ans, curr)
    
    return ans

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


def rotateRight(head: Optional[ListNode], k: int) -> Optional[ListNode]:

    if head is None:
        return None 

    n = 0
    cur_head = head
    while cur_head:
        n += 1
        prev = cur_head
        cur_head = cur_head.next
    
    prev.next = head # KEY: we have to make it a cycle
    k = k % n
    # k will be the new_head
    # n - k - 1 will be the new head
    # n - k = dist between head and new_head
    new_tail = head
    dist_new_head = n - k

    for _ in range(dist_new_head - 1):
        new_tail = new_tail.next
    
    head = new_tail.next
    new_tail.next = None

    return head

class Node:
    def __init__(self, key: int, val: int) -> None:
        # doubly linkedlist
        self.key = key
        self.val = val
        self.prev, self.next = None, None

class LRUCacheComplicatd:
    # interview use this
    # LRU --> get and put will be treated as used
    # remove least one in case key not in LRU_CACHE and size > capacity
    # otherwise we set lru_cache[key] = value
    # hashmap is for fast lookup O(1)
    # doublelinkedlist is for fast insert and deletion

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.lru_cache = dict() # key: Node
        self.left = Node(0, 0) # dummy
        self.right = Node(0, 0) # dummy
        self.left.next = self.right
        self.right.prev = self.left

    def remove(self, node):
        prev_node, next_node = node.prev, node.next
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def insert(self, node):
        prev_node, next_node = self.right.prev, self.right
        prev_node.next = node
        node.prev = prev_node
        node.next = next_node
        self.right.prev = node
        

    def get(self, key: int) -> int:
        if key in self.lru_cache:
            self.remove(self.lru_cache[key])
            self.insert(self.lru_cache[key])
            # emulate move_to_end
            return self.lru_cache[key].val
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.lru_cache:
            self.remove(self.lru_cache[key])
        self.lru_cache[key] = Node(key, value)
        self.insert(self.lru_cache[key])

        # remove LRU from the list
        # and key from hasmap
        if len(self.lru_cache) > self.capacity:
            lru = self.left.next
            self.remove(lru)
            del self.lru_cache[lru.key]


def reorderList(head: Optional[ListNode]) -> None:
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
    # head --> ... --> slow   null <-- slow ... <-- reversed_head 
    ori_head = head
    rev_head = reversed_head
    while rev_head.next: # very tricky !
        # since ori_head points to slow
        # the last node is fix
        # using rev_head will introduce a cycle
        # which cause infinite loop
        # null <-- 3 (stop at here)
        ori_next_node = ori_head.next
        rev_next_node = rev_head.next
    
        ori_head.next = rev_head
        rev_head.next = ori_next_node
    
        ori_head = ori_next_node
        rev_head = rev_next_node

def oddEvenList(head: Optional[ListNode]) -> Optional[ListNode]:

    if head is None:
        return None

    if head and not head.next:
        return head
    
    odd = head 
    even = head.next
    even_head = even

    # 4 in a row
    while even and even.next:
        odd.next = even.next
        odd = odd.next
        even.next = odd.next
        even = even.next
    
    odd.next = even_head
    return head 


def reverseList_recur(head: Optional[ListNode]) -> Optional[ListNode]:
    # 基本情況：如果是空鏈表或只有一個節點，直接返回
    if not head or not head.next:
        return head
    
    # 遞迴調用：先反轉後面的部分
    # newHead 將是原始鏈表的最後一個節點（反轉後的新頭節點）
    new_head = reverseList_recur(head.next)
    
    # 處理當前節點
    # head.next 是當前節點指向的下一個節點
    # 將下一個節點的 next 指向當前節點
    head.next.next = head
    
    # 斷開當前節點原本的 next 指針
    head.next = None

    return new_head

def reverseList_iter(head: Optional[ListNode]) -> Optional[ListNode]:
    prev = None
    curr = head
    while curr:
        next_temp = curr.next  # 保存下一個節點
        curr.next = prev      # 改變指針方向
        prev = curr          # 移動 prev
        curr = next_temp    # 移動 curr
    return prev


def firstMissingPositive(nums: list[int]) -> int:
    # cyclic sort

    n = len(nums)
    i = 0
    while i < n:
        # 1 at idx = 0
        # 2 at idx = 1
        act_pos = nums[i] - 1

        if 0 < nums[i] <= n and nums[i] != nums[act_pos]:
            nums[i], nums[act_pos] = nums[act_pos], nums[i]
        else:
            i += 1

    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    return n + 1


def maxSlidingWindow(nums: list[int], k: int) -> list[int]:
    # 239. Sliding Window Maximum
    # sliding window + monotonic stack

    queue = deque()
    ans = []

    for i in range(len(nums)):
        while queue and nums[queue[-1]] < nums[i]:
            queue.pop()

        queue.append(i)

        # out of k-bound
        if queue[0] <= i - k:
            queue.popleft()

        # finish k-bound
        if i >= k - 1:
            ans.append(nums[queue[0]])
    return ans


def reverseKGroup(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    if not head or not head.next or k == 1:
        return head
    
    dummy = ListNode(0)
    dummy.next = head
    group_prev = dummy

    while True:

        # check if k
        cur = group_prev
        for _ in range(k):
            cur = cur.next
            if not cur:
                return dummy.next

        cur_anchor = group_prev.next  # First node in the group
        next_node = cur_anchor.next  # Second node in the group

        for _ in range(k-1):  # Perform k-1 swaps to reverse the group
            cur_anchor.next = next_node.next  # Point current node to next's next
            next_node.next = group_prev.next  # Move `next_node` to the front
            group_prev.next = next_node  # Update group_prev's next to the new head
            next_node = cur_anchor.next  # Update next_node to the next in the group

        # Step 3: Move group_prev to the end of the reversed group
        group_prev = cur_anchor
        

def findWinners(matches: list[list[int]]) -> list[list[int]]:
    # 2225. Find Players With Zero or One Losses
    # couting sort 
    # edge case: 一個選手已經出現過且沒輸過
    lose_counts = [-1] * (100001)
    # + 1 because 1 <= winner_i, loser_i <= 1e5
    ans = [[], []]

    for winner, loser in matches:
        if lose_counts[winner] == -1:
            # this is the edge case
            # imagine the player only win !
            lose_counts[winner] = 0 
        if lose_counts[loser] == -1:
            lose_counts[loser] = 1
        else:
            lose_counts[loser] += 1
    
    for i in range(1, 100001): 
        if lose_counts[i] == 0:
            ans[0].append(i)
        if lose_counts[i] == 1:
            ans[1].append(i)
    return ans


def canConstruct(s: str, k: int) -> bool:
    # odd numbers dominate
    # one char: a, b
    # all even: abba, bb
    # 1 odd, others even:  aabcc

    if len(s) < k:
        return False
    
    hmp = dict()

    for char in s:
        hmp[char] = hmp.get(char, 0) + 1
    
    odd_count = 0
    for count in hmp.values():
        if count % 2:
            odd_count += 1

    return odd_count <= k


def minimumCardPickup(cards: list[int]) -> int:
    # 2260. Minimum Consecutive Cards to Pick Up

    hmp = dict()
    ans = float("inf")

    for idx, card in enumerate(cards):
        if card in hmp:
            ans = min(ans, idx - hmp[card] + 1)
        hmp[card] = idx

    return -1 if ans == float("inf") else ans


def maximumSum(nums: list[int]) -> int:
    # 2342. Max Sum of a Pair With Equal Sum of Digits
    hmp = dict()

    def get_digits_sum(num):
        acc = 0
        while num:
            acc += num % 10
            num //= 10
        return acc
    
    ans = float("-inf")
    for num in nums:
        ds = get_digits_sum(num)
        if ds not in hmp:
            hmp[ds] = num 
        else:
            # store max value of the ds
            # x + 最大值 一定比 x + 其他小值 要大
            # 第一行：拍照記錄當前找到的配對
            # 第二行：為之後的配對挑選最佳人選
            ans = max(ans, hmp[ds] + num)
            hmp[ds] = max(hmp[ds], num)
    return -1 if ans == float("-inf") else ans