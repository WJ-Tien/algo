from collections import Counter
from heapq import heappush, heappushpop, heappop

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
        return (-self.small[0] +self.large[0]) / 2
        

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
