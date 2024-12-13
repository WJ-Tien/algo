"""
詳細說明
Base case:

判斷當前路徑 path 是否滿足條件。如果滿足，將其加入結果中並返回。
例如在排列問題中，當 path 長度達到目標時，就是一個解。
選擇與撤銷選擇:

在迴圈中遍歷 options（可用的選項）。
選擇一個選項加入路徑，然後繼續遞歸。
當遞歸返回時，撤銷該選項（將其從路徑中移除），以便進行下一個選擇。
結果保存:

通常用一個全域變數 result 保存所有可能的解。
backtrack開始就是一個tree的 root
"""

def permute(nums: list[int]) -> list[list[int]]:
    # TC: O(N* N!) --> N depths * N! leaves
    # SC: O(N) --> recursion call stacks

    ans = []
    visited = [False] * len(nums) 
    def backtrack(path, visited):

        if len(path) == len(nums):
            ans.append(path[:])
            return
        
        for i in range(len(nums)):
            if visited[i]:
                continue
            path.append(nums[i]) 
            visited[i] = True
            backtrack(path, visited)
            path.pop()
            visited[i] = False
    backtrack([], visited)
    return ans

def combine(n: int, k: int) -> list[list[int]]:
    # [1..n]
    # TC: O(k * C(n, k)) --> k depth * C(n, k) leaves
    # SC: O(k) --> recursion call stacks

    ans = []
    def backtrack(start, path):
        if len(path) == k:
            ans.append(path[:])
            return

        for i in range(start, n + 1):
            path.append(i)
            backtrack(i + 1, path)
            path.pop()
    backtrack(1, []) 
    return ans


def subsets(nums: list[int]) -> list[list[int]]:
    # TC: O(n*2^n) --> n stack depth * 2^n results
    # SC: O(n)
    ans = []

    def backtrack(start, path, k):
        if len(path) == k:
            ans.append(path[:])
            return
        
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path, k)
            path.pop()
    
    for i in range(len(nums)+1):
        backtrack(0, [], i)
    return ans


def combinationSum(candidates: list[int], target: int) -> list[list[int]]:
    # n = len(candidates)
    # t is deepest when candidates = 1 (min(candidates))
    # TC: O(n^target) --> n, n^2, n^3 --> n^target
    # SC: O(t/min(candidates))

    ans = []
    def backtrack(path, start, cur_sum):
        if cur_sum == target:
            ans.append(path[:])
            return
        elif cur_sum > target:
            return
        
        for i in range(start, len(candidates)):
            path.append(candidates[i])
            backtrack(path, i, cur_sum + candidates[i]) 
            path.pop()
        
    backtrack([], 0, 0)
    return ans
