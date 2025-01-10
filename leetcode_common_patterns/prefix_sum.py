from collections import defaultdict

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