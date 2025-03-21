from collections import defaultdict

"""
nums[j+1] + ... + nums[i] = prefix_sum[i] - prefix_sum[j]
convert certain problems to prefix_sum
nums[j+1] + .. + nums[i] = some properties within this subarray

"""

def subarraySum(nums: list[int], k: int) -> int:
    """
    nums =    [1,  2,  3,  4]
    index:     0   1   2   3
    preSum:    1   3   6   10
    k = 5
    prefix[2] - prefix[0] = 6 - 1 = 5
    # curr - (curr-k) = k
    代表：nums[1] + nums[2] = 2 + 3 = 5
    prefix[j] = sum[0, j] #inclusive
    """

    # from index j to i
    # nums[j+1] + ... + nums[i] = preSum[i] - preSum[j]
    # nums[j] + ... + nums[i]  = preSum[i] - preSum[j-1]
    #                          = preSum[i] - preSum[j] + nums[j]
    # prefix[j] - prefix[i] = k 
    counts = defaultdict(int)
    counts[0] = 1 # prefix==k edge case k-k=0
    ans = curr = 0

    for num in nums:
        curr += num
        ans += counts[curr-k]
        counts[curr] += 1
    
    return ans 

    count = {0: 1}
    prefix_sum = 0
    ans = 0

    for idx, num in enumerate(nums):
        prefix_sum += num
        if prefix_sum - k in count:
            ans += count[prefix_sum - k]
        count[prefix_sum] = count.get(prefix_sum, 0) + 1 

    return ans

def numberOfSubarrays(nums: list[int], k: int) -> int:
    # odd counts
    counters = defaultdict(int)
    counters[0] = 1
    ans = 0
    curr = 0
    for num in nums:
        curr += num % 2
        ans += counters[curr - k]
        counters[curr] += 1

    return ans

    
def waysToSplitArray(nums: list[int]) -> int:
    # 2270. Number of Ways to Split Array

    ans = 0
    prefix = [nums[0]]

    for i in range(1, len(nums)):
        prefix.append(nums[i] + prefix[-1])
    
    for i in range(len(prefix)-1):
        if prefix[i] >= prefix[-1] - prefix[i]:
            ans += 1
    return ans

def waysToSplitArray_OPT(nums: list[int]) -> int:
    ans = 0
    total = sum(nums)
    left_sum = 0
    right_sum = total

    for i in range(len(nums)-1):
        left_sum += nums[i]
        right_sum -= nums[i]
        if left_sum >= right_sum:
            ans += 1
    return ans

def findMaxLength(nums: list[int]) -> int:
    # 525. contiguous array

    # if sum of a subarray = 0 --> same 0's and 1's

    prefix_sum = {0: -1} # prefix_sum : idx
    cur_sum = 0
    ans = 0

    for idx, num in enumerate(nums):
        if num == 0:
            cur_sum -= 1
        elif num == 1:
            cur_sum += 1
        
        if cur_sum in prefix_sum:
            # keep the oldest 
            # so that we can get longer length 
            # if cur_sum in prefix_sum
            # means from that idx
            # to the current idx
            # the sum of these items are zero
            # prefix[i] - prefix[j]
            # = nums[j+1] ~ nums[i] 
            # = 0
            ans = max(ans, idx - prefix_sum[cur_sum])
        else:
            prefix_sum[cur_sum] = idx
    return ans


def gridGame(grid: list[list[int]]) -> int:

    first_row_sum = sum(grid[0])
    second_row_sum = 0
    min_sum = float("inf")

    # NOTICE: second robot have to wait for first robot's action
    # So that's why we update second_row_sum after min_sum calc
    # the second robot either go top or bottom

    # 2 5 4
    # 1 5 1
    # turn at 0-idx
    # 0 5 4 first_row_sum = 5 + 4 = 9
    # 0 5 1 second_row_sum = 0 (not even started)
    # turn at 1-idx
    # 0 0 4 first_row_sum = 4
    # 1 0 1 second_row = 1


    for turn_idx in range(len(grid[0])):
        first_row_sum -= grid[0][turn_idx]
        min_sum = min(min_sum, max(first_row_sum, second_row_sum))
        second_row_sum += grid[1][turn_idx]
    
    return min_sum

def maxSubArrayLen(nums: list[int], k: int) -> int:

    max_len = float("-inf")
    prefix_sum = {0: -1}
    cur_sum = 0
    # edge case
    # 0 - (-1) = 1

    # prefix_sum[i] - prefix_sum[j] = nums[j+1] + ... + nums[i]
    # i - (j+1) + 1 = i - j

    for idx, num in enumerate(nums):
        cur_sum += num
        if cur_sum - k in prefix_sum:
            # i - (j+1) + 1 = i - j
            max_len = max(max_len, idx - prefix_sum[cur_sum - k])
        
        if cur_sum not in prefix_sum:
            prefix_sum[cur_sum] = idx
    
    return 0 if max_len == float("-inf") else max_len