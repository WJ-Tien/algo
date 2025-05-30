from collections import deque, defaultdict
from typing import Optional

class listNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def find_node(head, k):
    # find kth node from the end
    # using slow/fast pointers
    slow = head
    fast = head
    for _ in range(k):
        fast = fast.next
    
    while fast:
        slow = slow.next
        fast = fast.next
    
    return slow

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
    # 713. Subarray Product Less Than K

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
        
        # why left <= right ?
        # nums = [1, 1, 1]
        # k = 1

        ans += (right - left + 1) # key here !
        # (right-left+1) as the number of subarrays ending at "right" index
        # starting at left
        # [10, 5, 2]
        # @2 -> [2] & [5, 2]

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

def swapPairs(head: Optional[listNode]) -> Optional[listNode]:

    # the key is to use dummy node
    # while dummy.next will always be the same 
    # which points to the first group head
    # and prev to trace the new group head
    # if head is somehow dynamic
    # then you must use dummy node approach
    dummy = listNode(0)
    cur = dummy.next = head
    prev = dummy # to revise head

    # 1 -> 2 -> 3 -> 4
    while cur and cur.next:
        next_node = cur.next
        third_node = cur.next.next
        # 2 -> 1; 1-> 2
        next_node.next = cur
        # dummy -> 2 -> 1
        prev.next = next_node # critical
        # prev = last item before the reversed group 
        prev = cur # critical
        # dummy -> 2 -> 1 -> 3
        cur.next = third_node
        # move cur to the next group head
        cur = third_node
    return dummy.next

class RotateSol:

    def swap(self, nums, left, right):
        while left <= right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1

    def rotate(self, nums: list[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # [7,6,5,4,3,2,1]
            # [6,7] + [1,2,3,4,5]
            
        k = k % len(nums)

        # step 1: reverse all
        self.swap(nums, 0, len(nums)-1) 

        # step 2: reverse the first k
        # [0, k-1]
        self.swap(nums, 0, k-1)

        # step 3: reverse k+1 til the end
        #[k, :]
        self.swap(nums, k, len(nums)-1)

def rotateRight(self, head: Optional[listNode], k: int) -> Optional[listNode]:
    """
    1->2->3->4->5
    5->4->3->2->1
    4->5 
    1->2->3 
    4->5 -> 1->2->3
    """

    if not head or not head.next:
        return head

    list_len = self.get_list_len(head)
    k = k % list_len

    if k == 0:
        return head

    head, _ = self.rev_list(head, list_len)
    second_head = head
    for _ in range(k):
        second_head = second_head.next

    head, head_tail = self.rev_list(head, k)
    second_head, _ = self.rev_list(second_head, list_len - k)
    head_tail.next = second_head
    return head

def rev_list(self, head, k):
    prev = None
    cur = head
    for _ in range(k):
        next_node = cur.next
        cur.next = prev
        prev = cur
        cur = next_node
    return prev, head # new_head and tail

def get_list_len(self, head):
    cur = head
    list_len = 0
    while cur:
        list_len += 1
        cur = cur.next
    return list_len
    
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


def reorderlist(head: Optional[listNode]) -> None:
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

def oddEvenlist(head: Optional[listNode]) -> Optional[listNode]:

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


def reverselist_recur(head: Optional[listNode]) -> Optional[listNode]:
    # 基本情況：如果是空鏈表或只有一個節點，直接返回
    if not head or not head.next:
        return head
    
    # 遞迴調用：先反轉後面的部分
    # newHead 將是原始鏈表的最後一個節點（反轉後的新頭節點）
    new_head = reverselist_recur(head.next)
    
    # 處理當前節點
    # head.next 是當前節點指向的下一個節點
    # 將下一個節點的 next 指向當前節點
    head.next.next = head
    
    # 斷開當前節點原本的 next 指針
    head.next = None

    return new_head

def reverselist_iter(head: Optional[listNode]) -> Optional[listNode]:
    prev = None
    curr = head
    while curr:
        next_temp = curr.next  # 保存下一個節點
        curr.next = prev      # 改變指針方向
        prev = curr          # 移動 prev
        curr = next_temp    # 移動 curr
    return prev


def reverselist_dummy(head: Optional[listNode]) -> Optional[listNode]:

    dummy = listNode(0)
    cur = dummy.next = head
    new_head = dummy # key for using dummy node !
    prev = None

    while cur:
        next_node = cur.next
        cur.next = prev
        prev = cur
        cur = next_node
        new_head.next = prev # remember to update dummy/new_head
    
    return dummy.next


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


def reverseKGroup(head: Optional[listNode], k: int) -> Optional[listNode]:
    if not head or not head.next or k == 1:
        return head
    
    dummy = listNode(0)
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
    # 1400. Construct K Palindrome Strings
    # odd numbers dominate
    # one char: a, b
    # all even: abba, bb
    # 1 odd, others even:  aabcc
    """
    例如："aaaabbcc" 
    初始回文單位：["aa", "aa", "bb", "cc"]
    我們知道：
    每個回文單位都可以拆分成兩個更小的回文串
    每個回文單位也可以和其他單位合併
    而奇數一定要有搭配，或者是自己 --> very critical
    關鍵公式：
    如果目標是 k 個回文串：
    令 n = len(s)，n 一定是偶數（因為都是偶數次字符）
    令 p = 初始回文單位數量（每單位長度為2）
    所以 p = n/2
    我們可以得到：
    最少可以合併成 1 個回文串
    最多可以拆分成 n 個回文串（每個字符一個）--> very critical 
    因此：
    如果 1 ≤ k ≤ n，我們一定能通過適當的拆分/合併得到 k 個回文串
    這就是為什麼只需要檢查 k ≤ len(s)
    例如：n = 6 的情況
    Copys = "aaabbb"
    1個回文串：   ["aaabbb"]
    2個回文串：   ["aaa", "bbb"]
    3個回文串：   ["aa", "a", "bbb"]
    4個回文串：   ["aa", "a", "bb", "b"]
    5個回文串：   ["a", "a", "a", "bb", "b"]
    6個回文串：   ["a", "a", "a", "b", "b", "b"]
    這就是完整的數學解釋！任何 k 在 [1, n] 範圍內，我們都能構造出對應數量的回文串。 CopyRetryClaude can make mistakes. Please double-check responses.
    """

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



def countServers(grid: list[list[int]]) -> int:
    # 1267. Count Servers that Communicate
    # T: O(m*n)
    # S: O(m+n)

    # array grouping
    rows, cols = len(grid), len(grid[0])
    row_counts = [0] * rows
    col_counts = [0] * cols
    ans = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                row_counts[r] += 1
                col_counts[c] += 1
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and (row_counts[r] > 1 or col_counts[c] > 1):
                ans += 1

    return ans


def lexicographicallySmallestArray(nums: list[int], limit: int) -> list[int]:
    # 2948. Make Lexicographically Smallest Array by Swapping Elements
    # [1, 2, 1] < [1, 2, 2] # compare num by idx
    # 1 == 1,  2 == 2, 1 < 2 
    # [1,2,1] < [1,2,2]
    n = len(nums)
    # 將數值和位置配對並排序
    pairs = sorted((num, i) for i, num in enumerate(nums))
    
    groups = []
    curr_group = [pairs[0]]
    
    # 分組：差距不超過 limit 的數字歸為一組
    for i in range(1, n):
        if pairs[i][0] - pairs[i-1][0] <= limit:
            curr_group.append(pairs[i])
        else:
            groups.append(curr_group)
            curr_group = [pairs[i]]
    groups.append(curr_group)
    
    # 建立結果陣列
    result = [0] * n
    
    # 處理每個群組
    for group in groups:
        # 獲取位置和數值
        positions = sorted(p[1] for p in group)
        values = sorted(p[0] for p in group)
        
        # 按照排序後的位置放入對應的值
        for pos, val in zip(positions, values):
            result[pos] = val
    
    return result


def removeDuplicates_I(nums: list[int]) -> int:
    # 26. Remove Duplicates from Sorted Array
    # two pointers approach
    slow = 0

    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    return slow + 1 # idx + 1 == len


def removeDuplicates_I_better(nums: list[int]) -> int:

    k = 1
    if len(nums) < k:
        return len(nums)
    
    slow = k

    for fast in range(k, len(nums)):
        if nums[fast] != nums[slow-k]:
            nums[slow] = nums[fast]
            slow += 1
    return slow

def removeDuplicates_II(nums: list[int]) -> int:
    # 80. Remove Duplicates from Sorted Array II
    # < k --> swap in ; >= k, discard
    # always check window size == k

    k = 2
    if len(nums) < k:
        return len(nums)

    slow = k

    for fast in range(k, len(nums)):
        if nums[fast] != nums[slow-k]:
            nums[slow] = nums[fast]
            slow += 1
    return slow

def merge(nums1: list[int], m: int, nums2: list[int], n: int) -> None:
    # 88. Merge Sorted Array
    """
    Do not return anything, modify nums1 in-place instead.
    """
    # nums1 = m + n
    p1 = m - 1
    p2 = n - 1
    p = m + n - 1

    while p1 >= 0 and p2 >= 0:
        if nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
        p -= 1
    
    while p2 >= 0:
        nums1[p] = nums2[p2]
        p2 -= 1
        p -= 1

    while p1 >= 0:
        nums1[p] = nums1[p1]
        p1 -= 1
        p -= 1


class Logger:
    # 359. Logger Rate Limiter
    # hashmap --> deque (better memory effiency)

    def __init__(self):
        # dict[message, timestamp]
        # T: O(1)
        # S: O(N)
        self.check_msg = set() 
        self.msgq = deque() # [msg, timestamp]

    def shouldPrintMessage(self, timestamp: int, message: str) -> bool:

        # remove all items <= timestamp-10 to keep memory-efficient

        while self.msgq:
            old_msg, old_timestamp = self.msgq[0]
            if timestamp - old_timestamp >= 10:
                self.msgq.popleft()
                self.check_msg.remove(old_msg)
            else:
                break

        if message not in self.check_msg:
            self.check_msg.add(message)
            self.msgq.append((message, timestamp))
            return True
        else:
            return False 
    
    # def __init__(self):
    #     # dict[message, timestamp]
    #     self.hmp = dict()

    # def shouldPrintMessage(self, timestamp: int, message: str) -> bool:
    #     if message not in self.hmp:
    #         self.hmp[message] = timestamp + 10
    #         return True
    #     else:
    #         if timestamp < self.hmp[message]:
    #             return False
    #         self.hmp[message] = timestamp + 10
    #         return True


def findDisappearedNumbers(nums: list[int]) -> list[int]:

    for i in range(len(nums)):

        idx = abs(nums[i]) - 1

        if nums[idx] > 0:
            nums[idx] *= -1
    
    ans = []
    for i in range(1, len(nums)+1):
        if nums[i-1] > 0:
            ans.append(i)
    return ans


def containsNearbyDuplicate(nums: list[int], k: int) -> bool:
    # 219. Contains Duplicate II

    # T: O(n)
    # S: O(min(n, k))
    # n: false, k: true

    # hashmap + sliding window
    hmp = dict()

    for idx, num in enumerate(nums):
        if num in hmp and idx - hmp[num] <= k:
            return True
        hmp[num] = idx
    return False


def deleteDuplicates(head: Optional[listNode]) -> Optional[listNode]:
    # 83. Remove Duplicates from Sorted list

    if head is None:
        return head
    
    prev = head
    cur = head.next
    while cur:
        if cur.val == prev.val:
            prev.next = cur.next
        else:
            prev = prev.next
        cur = cur.next
    return head



def firstUniqChar(s: str) -> int:

    hmp = dict()
    uni = []

    for idx, char in enumerate(s):
        if char not in hmp:
            uni.append((char, idx))
            hmp[char] = 1
        else:
            hmp[char] += 1
    
    for char, idx in uni:
        if hmp[char] == 1:
            return idx 
    return -1


def maxProfit(prices: list[int]) -> int:
    # 122. Best Time to Buy and Sell Stock II
    # 7, 1, 5, 3, 6, 4
    # sum every nearest valley and peak (buy at low, sell at high)
    # i.e. every positive profit
    # 6 - 1 = 5 is not the max_profit
    # instead, 5-1 + 6-3 = 7 is 
    max_profit = 0

    for i in range(len(prices)-1):
        net = prices[i+1] - prices[i]
        if net > 0:
            max_profit += net
    return max_profit


def minimumRecolors(blocks: str, k: int) -> int:
    # 2379. Minimum Recolors to Get K Consecutive Black Blocks
    left = 0
    white = 0
    ans = float("inf")

    for right in range(len(blocks)):
        if blocks[right] == "W":
            white += 1
        
        # sliding window with size k
        if right - left + 1 == k:
            ans = min(ans, white)

            if blocks[left] == "W":
                white -= 1
            left += 1
    return ans


def canPlaceFlowers(f, flowerbed: list[int], n: int) -> bool:
    # 0 + flowerbed + 0 for edge cases
    # [0, 0, 0] or [0, 0, 1]
    # O(n) TS
    new_flowerbed = [0] + flowerbed + [0]
    x = n

    for i in range(1, len(new_flowerbed)-1):
        if new_flowerbed[i-1] == 0 and new_flowerbed[i] == 0 and new_flowerbed[i+1] == 0:
            new_flowerbed[i] = 1
            x -= 1 
    return x <= 0

    def canPlaceFlowers_opt(flowerbed: list[int], n: int) -> bool:

            cur_cnt = 0
            idx = 0

            while idx < len(flowerbed):
                if flowerbed[idx] == 0:
                    # left [0, 0, 1] or [1, 0, 0, 0]
                    prev_empty = (idx == 0 or flowerbed[idx-1] == 0)
                    # right [1, 0, 0] or [0, 0, 0, 1]
                    next_empty = (idx == len(flowerbed)-1 or flowerbed[idx+1] == 0)

                    if prev_empty and next_empty:
                        flowerbed[idx] = 1
                        cur_cnt += 1
                        if cur_cnt >= n:
                            return True
                        
                        idx += 1 # important
                idx += 1

            # edge case [1, 0, 1, 0], n = 0 
            # so we can't return False here
            return cur_cnt >= n

def removeElements(head: Optional[listNode], val: int) -> Optional[listNode]:

    dummy = listNode(0, head)  # Dummy node simplifies removal logic
    cur = dummy

    while cur.next:
        if cur.next.val == val:
            cur.next = cur.next.next  # remove node by skipping it
        else:
            cur = cur.next  # move to next node if no removal

    return dummy.next


def maximumTripletValue(nums: list[int]) -> int:
    # 2873. Maximum Value of an Ordered Triplet I
    # 如果我把陣列從左到右遍歷，每次把當前數字當作 nums[k]，
    # 那麼問題就變成：在 k 左邊的所有數字中，
    # 找到一個最大的 (nums[i] - nums[j])，然後乘以當前的 nums[k]。
    # nums[i] - nums[j] 找max_diff
    # nums[i] 找max_num

    ans = 0
    max_diff = 0
    max_num = 0

    for num in nums:
        ans = max(ans, max_diff * num) # k
        max_diff = max(max_diff, max_num - num) # j
        max_num = max(max_num, num) # i
    
    return ans


def isHappy(self, n: int) -> bool:
    # 202. Happy Number

    slow = n
    fast = self.square_sum_of_each_digit(n)
    while fast != 1 and fast != slow:
        slow = self.square_sum_of_each_digit(slow)
        fast = self.square_sum_of_each_digit(self.square_sum_of_each_digit(fast))
    return fast == 1
    def square_sum_of_each_digit(self, n: int) -> int:
        cur_sum = 0
        while n:
            digit = n % 10
            cur_sum += digit**2
            n //= 10
        return cur_sum

    class Solution:
        def isHappy(self, n: int) -> bool:

            x = n
            check = set()
            check.add(x)
            while x != 1:
                x = self.square_sum_of_each_digit(x)
                if x in check:
                    return False
                check.add(x)
            
            return True
        def square_sum_of_each_digit(self, n: int) -> int:
            cur_sum = 0
            while n:
                digit = n % 10
                cur_sum += digit**2
                n //= 10
            return cur_sum

        
def replaceElements(arr: list[int]) -> list[int]:
    # 1299. Replace Elements with Greatest Element on Right Side
    # just reverse it !
    cur_max = float("-inf")
    ans = [-1] * len(arr)
    for i in range(len(arr)-1, -1, -1): 
        if cur_max == float("-inf"):
            ans[i] = -1
        else:
            ans[i] = cur_max
        cur_max = max(cur_max, arr[i])
    return ans


def numberOfPoints(nums: list[list[int]]) -> int:

    # 2848. Points That Intersect With Cars
    # diff method
    diff = [0] * 102

    for start, end in nums:
        diff[start] += 1
        diff[end+1] -= 1

    n_points = 0 
    car_cumsum = 0
    for i in range(len(diff)):
        car_cumsum += diff[i]
        if car_cumsum > 0:
            n_points += 1
    
    return n_points


def minimumOperations(nums: list[int]) -> int:
    seen = [False] * 101
    for i in range(len(nums) - 1, -1, -1):
        if seen[nums[i]]:
            return i // 3 + 1
        seen[nums[i]] = True
    return 0



def zeroFilledSubarray(self, nums: list[int]) -> int:

    ans = 0
    run = 0

    for num in nums:
        if num == 0:
            run += 1
        else:
            ans += (run+1)*(run) // 2
            run = 0

    # edge case
    if run > 0:
        ans += (run+1)*(run) // 2
    return ans


def isSubsequence(self, s: str, t: str) -> bool:
    # 392. Is Subsequence

    # two pointers
    # compare idx !

    # T: O(T)
    # S: O(1)

    s_bound, t_bound = len(s), len(t)

    ptr_s = ptr_t = 0

    while ptr_s < s_bound and ptr_t < t_bound:
        if s[ptr_s] == t[ptr_t]:
            ptr_s += 1
        ptr_t += 1
    
    return ptr_s == s_bound


def increasingTriplet(self, nums: list[int]) -> bool:

    # 334. Increasing Triplet Subsequence

    first = float("inf")
    second = float("inf")

    for num in nums:
        if num <= first:
            first = num
        elif num <= second:
            second = num
        else:
            return True
    
    return False



def countQuadruplets(self, nums: list[int]) -> int:
    # 1995. Count Special Quadruplets
    # nums[a] + nums[b] = nums[d] - nums[c]

    n = len(nums)
    ans = 0
    diffCnt = defaultdict(int)  

    for b in range(n - 3, 0, -1):
        c = b + 1 
        for d in range(c + 1, n):
            diff = nums[d] - nums[c]
            diffCnt[diff] += 1

        for a in range(b):          
            need = nums[a] + nums[b]
            ans += diffCnt.get(need, 0) 

    return ans


def fourSum(self, nums: list[int], target: int) -> list[list[int]]:
    
    nums.sort()                 # 先排序，方便雙指標收斂
    n = len(nums)
    res = []

    for i in range(n - 3):      # 第一層固定 i
        if i > 0 and nums[i] == nums[i - 1]:
            continue            # 跳過重複 i 值

        for j in range(i + 1, n - 2):   # 第二層固定 j
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue        # 跳過重複 j 值

            left, right = j + 1, n - 1  # 內層雙指標
            while left < right:
                s = nums[i] + nums[j] + nums[left] + nums[right]
                if s == target:
                    res.append([nums[i], nums[j], nums[left], nums[right]])
                    left += 1
                    right -= 1
                    # 去重：連跳相同 left / right 值
                    while left < right and nums[left] == nums[left - 1]:
                        left += 1
                    while left < right and nums[right] == nums[right + 1]:
                        right -= 1
                elif s < target:        # 總和太小 → 擴大 left
                    left += 1
                else:                   # 總和太大 → 縮小 right
                    right -= 1
    return res


# =========================================================================================

def fourSum_General(self, nums: list[int], target: int) -> list[list[int]]:
    # T: O(n^(k-1))
    # S: O(n)
    return self.kSum(nums, 4, target)

def kSum(self, nums: list[int], k: int, target: int) -> list[list[int]]:
    nums.sort()                         # 先排序，方便去重與雙指標
    return self._kSum(nums, 0, k, target)

# 內部遞迴：從 start 開始找 k 個數，使其和為 target
def _kSum(self, nums: list[int], start: int, k: int, target: int) -> list[list[int]]:
    res = []
    n = len(nums)

    # ===== 基底：2-Sum，直接雙指標 =====
    if k == 2:
        left, right = start, n - 1
        while left < right:
            s = nums[left] + nums[right]
            if s == target:
                res.append([nums[left], nums[right]])
                left += 1
                right -= 1
                # ---- 去重（兩端都跳）----
                while left < right and nums[left] == nums[left - 1]:
                    left += 1
                while left < right and nums[right] == nums[right + 1]:
                    right -= 1
            elif s < target:
                left += 1
            else:                       # s > target
                right -= 1
        return res

    # ===== 遞迴層：固定一個 nums[i]，往下找 (k-1)-Sum =====
    for i in range(start, n - k + 1):
        # -- 去重：同值的 i 只處理一次
        if i > start and nums[i] == nums[i - 1]:
            continue
        # 往下遞迴求子組合
        for subset in self._kSum(nums, i + 1, k - 1, target - nums[i]):
            res.append([nums[i]] + subset)

    return res

# =========================================================================================