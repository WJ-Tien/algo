
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
