"""
關鍵區別：為什麼 57 需要 max 而 56 不需要？
LeetCode 57：需要 max 的原因
因為我們是在處理一個新的插入區間，它的起點和終點是未知的，且可能會與多個區間重疊。

我們需要確保找到所有重疊區間的最大終點，才能確定插入後的正確範圍。
這是因為插入的區間可能擴展現有區間的終點。
LeetCode 56：不需要額外考慮起點
在 LeetCode 56 中，區間已經排好序，因此每個區間的起點只可能在前一區間的右側或完全包含於前一區間內。

起點已經在排序過程中保證是最小的。
我們只需要關注合併後的結束點是否需要延伸。
簡要總結：
LeetCode 57 中，我們處理「一個新的未確定的區間」，需要動態更新 起點（min）和終點（max）。
LeetCode 56 中，所有區間已經排序好，只需要關注合併後的 終點（max），起點已經由排序保證是最小的。


"""
def merge(intervals: list[list[int]]) -> list[list[int]]:

    intervals.sort(key=lambda x: x[0])
    ans = []
    cur_interval = intervals[0]

    for i in range(1, len(intervals)):
        # overlapped
        # [1 (2 3] 4) --> cur_interval = [1 4]
        if is_overlapped(cur_interval, intervals[i]):
            cur_interval[1] = max(intervals[i][1], cur_interval[1])
        
        # [1 4] [5 6] --> cur_interval = [5 6]
        else:
            ans.append(cur_interval)
            cur_interval = intervals[i]

    ans.append(cur_interval)

    return ans


def insert(intervals: list[list[int]], newInterval: list[int]) -> list[list[int]]:

    # three parts
    # 1. non-overlapped left
    # 2. overlapped
    # 3. non-overlapped right

    i = 0
    n = len(intervals)
    ans = [] 

    # [1 2] [3 5]
    while i < n and intervals[i][1] < newInterval[0]:
        ans.append(intervals[i])
        i += 1
    
    # [3 (4 5] 6)
    # min --> new start
    # max --> new end
    # iteratively update newInterval !
    while i < n and is_overlapped(intervals[i], newInterval):
        newInterval[0] = min(intervals[i][0], newInterval[0])
        newInterval[1] = max(intervals[i][1], newInterval[1])
        i += 1
    ans.append(newInterval)

    # [5 6][7 8]
    while i < n:
        ans.append(intervals[i])
        i += 1
    
    return ans

def is_overlapped(interval1: list[list[int]], interval2: list[list[int]]) -> bool:
    # no overlapped 
    if (interval1[0] > interval2[1]) or (interval2[0] > interval1[1]):
        return False
    # overlapped 
    return True


def eraseOverlapIntervals(intervals: list[list[int]]) -> int:

    intervals.sort(key=lambda x: x[0])
    ans = 0
    prev_end = intervals[0][1]

    for start, end in intervals[1:]:
        # not-overlapped
        if start >= prev_end:  
            prev_end = end
        else:
            ans += 1
            prev_end = min(prev_end, end)
            # greedy way
            # the earlier we keep the smaller end instead of larger end
            # the more likely we can remove less intervals
    return ans

class Interval:
    def __init__(self, start: int = None, end: int = None):
        self.start = start
        self.end = end


def employeeFreeTime(schedule: list[list[Interval]]) -> list[Interval]:
    # O(NlogN))
    # O(N)
    # refer to merge intervals

    intervals = []
    for employee in schedule:
        for interval in employee:
            intervals.append([interval.start, interval.end])
    
    intervals.sort(key=lambda x: x[0])


    merge = [intervals[0]]

    for interval in intervals[1:]:
        if merge[-1][1] >= interval[0]:
            merge[-1][1] = max(merge[-1][1], interval[1])
        else:
            merge.append(interval)
    
    ans = []
    for i in range(1, len(merge)):
        if merge[i][0] > merge[i-1][1]:
            # key: append Interval not arrays/lists !!!
            ans.append(Interval(merge[i-1][1], merge[i][0]))
    return ans 