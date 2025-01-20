from collections import Counter
from heapq import heappush, heappushpop, heappop
from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Python itself implements "min_heap"

# heap sorted by the first element of the tuple
# and then by the second element and etc
# so it's always a good idea to store data in a tuple
# or we can 
# use customized class to redefine __lt__ 
# (please see top K elements)
# heap only guarantees to store the smallest value at the first node
# and recursively to the substree root
# while the others are not sorted

"""
insert num to a 
1. max_heap: -num
2. min_heap:  num
"""

def kClosest(points: list[list[int]], k: int) -> list[list[int]]:

    hp = []
    ans = []

    for (px, py) in points:

        dist = -(px**2 + py**2)
        if len(hp) == k:
            heappushpop(hp, (dist, [px, py]))
        
        else:
            heappush(hp, (dist, [px, py]))
    
    # no guaranteed sorted
    for i in range(k):
        ans.append(hp[i][1])

    return ans


def findClosestElements(arr: list[int], k: int, x: int) -> list[int]:
    # optimal: sliding window + binary search
    hp = []

    for num in arr:
        dist = -abs(num - x)
        if len(hp) == k:
            heappushpop(hp, (dist, -num))
        else:
            heappush(hp, (dist, -num))

    return sorted([-hp[i][1] for i in range(k)])


# Really interesting 
# Custom class for heap elements
class WordFrequency:
    def __init__(self, word, frequency):
        self.word = word
        self.frequency = frequency
    
    def __lt__(self, other):
        # Compare based on frequency first (ascending because it's a min-heap)
        if self.frequency == other.frequency:
            # If frequencies are the same, compare lexicographically
            return self.word > other.word  # Reverse lexicographical order for min-heap
        return self.frequency < other.frequency  # Min-heap by frequency

class Solution:

    def topKFrequent(self, words, k):
        # Step 1: Count word frequencies
        freq_map = Counter(words)
    
        # Step 2: Use a heap to keep the top k elements
        heap = []
        for word, freq in freq_map.items():

            if len(heap) == k:
                heappushpop(heap, WordFrequency(word, freq))
            else:
                heappush(heap, WordFrequency(word, freq))
    
        # Step 3: Extract elements from the heap
        ans = []
        for i in range(k):
            ans.append(heappop(heap).word)
    
        # Reverse the result since we want the most frequent first
        return ans[::-1]


class MedianFinder:

    def __init__(self):
        # T: O(logN)
        # S: O(N)
        # 1 2(max from max_heap) 3(min from min_heap) 4
        self.small = [] # store smallers', max_heap
        self.large = [] # store largers', min_heap
        
    def addNum(self, num: int) -> None:
        heappush(self.small, -num)
        if self.small and self.large:
            # small always <= large
            if -self.small[0] > self.large[0]:
                val = -heappop(self.small)
                heappush(self.large, val)
        
        # balance heaps
        if len(self.small) > len(self.large) + 1:
            val = -heappop(self.small)
            heappush(self.large, val)
        if len(self.small) + 1 < len(self.large):
            val = heappop(self.large)
            heappush(self.small, -val)
        
    def findMedian(self) -> float:
        # 1 2 3 . 4 5 
        if len(self.small) > len(self.large):
            return -self.small[0]
        # 1 2 . 3 4 5
        if len(self.small) < len(self.large):
            return self.large[0]
        # 1 2 . 3 4
        return (-self.small[0] + self.large[0]) / 2
        

def minMeetingRooms(intervals: list[list[int]]) -> int:
    # O(n*logn)
    # O(n)

    # save end to the heap
    mhp = []
    intervals.sort(key=lambda x: x[0])

    for start, end in intervals:
        if mhp and mhp[0] <= start: # end before new -> pop it
            heappop(mhp)
        heappush(mhp, end) # push end
    
    return len(mhp)


def mergeKLists(lists: list[Optional[ListNode]]) -> Optional[ListNode]:
    new_head = None 
    cur_head = new_head
    hp = []

    # only keep k items in the hp
    # since we can utilize the asc nature
    for idx, cur_list in enumerate(lists):
        if cur_list:
            heappush(hp, (cur_list.val, idx, cur_list))
    
    while hp: 
        val, idx, cur_list = heappop(hp)
        if new_head is None:
            new_head = ListNode(val)
            cur_head = new_head
        else:
            cur_head.next = ListNode(val)
            cur_head = cur_head.next

        if cur_list.next:
            heappush(hp, (cur_list.next.val, idx, cur_list.next))

    return new_head

class Interval:
    def __init__(self, start: int = None, end: int = None):
        self.start = start
        self.end = end

def employeeFreeTime(schedule: list[list[Interval]]) -> list[Interval]:
    # merge intervals + merge k sorted lists
    # O(Nlogk)
    # O(N)
    hp = []

    for emp_id, employee in enumerate(schedule):
        heappush(hp, (employee[0].start, employee[0].end, emp_id, 0))
        # 0 = employee 0's schedule 
    
    ans = []
    prev_end = None
    while hp:
        start, end, emp_idx, interval_idx = heappop(hp)

        if prev_end is not None and start > prev_end:
            ans.append(Interval(prev_end, start))

        if prev_end is not None:
            prev_end = max(prev_end, end)
        else:
            prev_end = end
        
        if interval_idx + 1 < len(schedule[emp_idx]):
            next_interval = schedule[emp_idx][interval_idx + 1]
            heappush(hp, (next_interval.start, next_interval.end, emp_idx, interval_idx+1))
    return ans

def smallestRange(nums: list[list[int]]) -> list[int]:
    # min_heap + sliding window
    """
    A = [1, 4, 7]
    B = [2, 5, 8]
    我們現在有 1 和 2
    如果我們移動 2（較大的數），新範圍會是 [1, 5]，差距變大了
    如果我們移動 1（較小的數），新範圍會是 [2, 4]，這可能會更好
    這就是為什麼我們總是移動最小的數字！因為：
    保持最小值不變，範圍只會變得更大
    移動較大的值，範圍一定會變大

    """

    hp = []
    max_value = float("-inf")

    for i in range(len(nums)):
        # hp: value, num_idx, n_th element in num[i]
        heappush(hp, (nums[i][0], i, 0))
        max_value = max(max_value, nums[i][0])
    
    start, end = float("-inf"), float("inf")

    while hp:
        # this is moving min to the left
        # sliding window like mechanism
        min_value, num_idx, i = heappop(hp)
        if max_value - min_value < end - start:
            start, end = min_value, max_value
        
        if i + 1 == len(nums[num_idx]):
            break
        
        
        if i + 1 < len(nums[num_idx]): # non-empty
            next_value = nums[num_idx][i+1]
            heappush(hp, (next_value, num_idx, i+1))
            max_value = max(max_value, next_value)
    
    return [start, end]


def trapRainWater(heightMap: list[list[int]]) -> int:
    # 407. Trapping Rain Water II

    m, n = len(heightMap), len(heightMap[0])
    
    # 初始化訪問數組和優先隊列
    visited = [[False] * n for _ in range(m)]
    heap = []
    
    # 將邊界放入優先隊列
    # 添加第一行和最後一行
    for j in range(n):
        heappush(heap, (heightMap[0][j], 0, j))
        heappush(heap, (heightMap[m-1][j], m-1, j))
        visited[0][j] = visited[m-1][j] = True
    
    # 添加第一列和最後一列
    for i in range(1, m-1):
        heappush(heap, (heightMap[i][0], i, 0))
        heappush(heap, (heightMap[i][n-1], i, n-1))
        visited[i][0] = visited[i][n-1] = True
    
    # 四個方向的移動
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    result = 0
    max_height = 0
    
    # 從外向内遍歷
    while heap:
        height, row, col = heappop(heap)
        max_height = max(max_height, height)
        
        # 檢查四個相鄰的格子
        for dx, dy in directions:
            new_row, new_col = row + dx, col + dy
            
            # 檢查是否在邊界內且未訪問
            if (0 <= new_row < m and 0 <= new_col < n 
                and not visited[new_row][new_col]):
                
                visited[new_row][new_col] = True
                curr_height = heightMap[new_row][new_col]
                
                # 如果當前格子高度小於最大高度，可以積水
                if curr_height < max_height:
                    result += max_height - curr_height
                
                heappush(heap, (curr_height, new_row, new_col))
    
    return result