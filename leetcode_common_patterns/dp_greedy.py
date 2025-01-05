from bisect import bisect_right

def maximalSquare(matrix: list[list[str]]) -> int:

    # dp[r][c] = min(left, top, diag)

    rows, cols = len(matrix), len(matrix[0])
    dp = [[0] * cols for _ in range(rows)]
    ans = 0

    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == "1":
                dp[r][c] = 1
                if r > 0 and c > 0:
                    dp[r][c] += min(dp[r-1][c], dp[r][c-1], dp[r-1][c-1])
                ans = max(ans, dp[r][c])
                
    return ans * ans

    
def coinChange(coins: list[int], amount: int) -> int:
    # greedy algo
    dp = [float("inf")] * (amount + 1) 
    dp[0] = 0

    for coin in coins:
        for x in range(coin , amount + 1):
            # x = x - "coin" and add one "coin"
            # dp[x] = min steps to make up x
            dp[x] = min(dp[x], dp[x - coin] + 1)

    return dp[amount] if dp[amount] != float("inf") else -1


def numDecodings(s: str) -> int:
    # 91 Decode ways
    # dp[i] 表示到第i個字符為止，可以解碼的方法數量
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1 # empty string
    dp[1] = 1 # 1

    if s == "" or s[0] == '0':
        return 0

    for i in range(2, n+1):
        if s[i-1] != '0':
            dp[i] += dp[i-1]
        
        two_digit = int(s[i-2: i]) 
        if 10 <= two_digit <= 26:
            dp[i] += dp[i-2]

    return dp[len(s)]

        
def canJump(nums: list[int]) -> bool:
    max_reach = 0

    for idx, num in enumerate(nums):
        if idx > max_reach:
            return False
        
        max_reach = max(max_reach, idx + num)

        if max_reach >= len(nums) - 1:
            return True
    return False


def canPartition(nums: list[int]) -> bool:
    # 416. Partition Equal Subset Sum
    # compared with Coin change

    target = sum(nums)
    if target % 2:
        return False
    target //= 2

    dp = [False] * (target + 1)
    dp[0] = True

    for num in nums:
        for i in range(target, num-1, -1):
            dp[i] = dp[i] or dp[i - num]
    return dp[target]


def wordBreak(s: str, wordDict: list[str]) -> bool:

    # T: O(n * m * k)
    # S: O(n)

    n = len(s)
    dp = [False] * (n+1)
    dp[0] = True


    for i in range(n):
        if not dp[i]:
            continue
        
        for word in wordDict:
            if s[i:].startswith(word):
                dp[i + len(word)] = True
    return dp[n]


"""
外層遍歷硬幣 → Coin Change
避免重複計算相同的組合。
保證不考慮排列順序，只關注最優解。

外層遍歷目標金額 → Combination Sum IV
允許考慮排列順序。
確保每個金額可以累計所有排列數量。
"""
def combinationSum4(nums: list[int], target: int) -> int:

    dp = [0] * (target + 1)
    dp[0] = 1

    for t in range(1, target+1):
        for num in nums:
            if t >= num:
                dp[t] += dp[t-num]
    return dp[target]



# greedy
def canCompleteCircuit(gas: list[int], cost: list[int]) -> int:
    """
    為什麼需要更新到 i + 1？ 當前油量不足，無法繼續?
    
    如果在第 i 個加油站時，current_tank < 0，
    說明從目前的起點一直到第 i 個加油站的累計油量無法支撐到達下一個加油站。
    這意味著從當前起點 start 到第 i 個加油站的所有加油站，都不可能作為有效的起點，因為：
    如果 start 到 
    i 這段路都無法繼續，那麼任何起點在這段路中間的情況下，
    剩餘油量只會更少，依然無法到達。
    下一站可能有更好的機會：
    當 current_tank < 0 時，我們直接跳到下一個加油站 
    i+1，將其作為新的起點，重新開始累積油量。
    這是因為剩下的油量不足以到達下一個加油站，因此嘗試從 
    i+1 開始重新累積油量。
    """
    total_tank = 0
    cur_tank = 0
    start = 0

    for i in range(len(gas)):
        total_tank += gas[i] - cost[i]
        cur_tank += gas[i] - cost[i]

        if cur_tank < 0:
            cur_tank = 0
            start = i + 1
    return start if total_tank >= 0 else -1


def jobScheduling(startTime: list[int], endTime: list[int], profit: list[int]) -> int:
    # T: O(NlogN)
    # S: O(N)
    jobs = sorted(zip(startTime, endTime, profit), key=lambda x: x[1])
    dp = [(0, 0)] # end, cur_max_profit
    
    # dp[i] 表示考慮到第 i 個工作時，最大可獲得的利潤。
    for start, end, profit in jobs:
        # ol_end --> new_start
        idx = bisect_right(dp, (start, float("inf"))) - 1
        if idx < 0:
            continue

        cur_profit = dp[idx][1] + profit

        if cur_profit > dp[-1][1]:
            dp.append((end, cur_profit))
    
    return dp[-1][1]