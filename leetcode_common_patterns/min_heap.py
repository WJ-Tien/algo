from collections import Counter
from heapq import heappush, heappushpop, heappop

# heap sorted by the first element of the tuple
# and then by the second element and etc
# so it's always a good idea to store data in a tuple
# or we can 
# use customized class to redefine __lt__ 
# (please see top K elements)
# heap only guarantees to store the smallest value at the first node
# while the others are not sorted

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
    