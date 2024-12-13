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

"""

def permute(nums: list[int]) -> list[list[int]]:
    # TC: O(N* N!) --> N depths * N! leaves
    # SC: O(N) --> recursion call stacks

    ans = []
    visited = [False] * len(nums) 
    def backtrack(path, visited):

        if len(path) == len(nums):
            ans.append(path[:])
        
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

        for i in range(start, n + 1):
            path.append(i)
            backtrack(i + 1, path)
            path.pop()
    backtrack(1, []) 
    return ans






def subsets(nums):
    def backtrack(start, path):
        result.append(path[:])  # 每次都加入當前子集
        for i in range(start, len(nums)):
            path.append(nums[i])  # 選擇
            backtrack(i + 1, path)  # 遞歸下一步
            path.pop()  # 撤銷選擇

    result = []
    backtrack(0, [])
    return result
