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


def generateParenthesis(n: int) -> list[str]:
    # TC: O(4^n/sqrt(n)) (O(2^n))
    # SC: O(n)
    
    ans = []

    def backtrack(left, right, path):
        if left == n and right == n:
            ans.append(''.join(path[:]))
            return
    
        if left < n:
            path.append("(")
            backtrack(left + 1, right, path)
            path.pop()
        if right < left:
            path.append(")")
            backtrack(left, right + 1, path)
            path.pop()

    backtrack(0, 0, [])
    return ans 



def letterCombinations(digits: str) -> list[str]:
    # TC: O(4^N * (N+N)) == O(4^N*N). 4^N results * (N recursion stack + N join)
    # SC: O(N + 4^N * N)
    # Rule of product 
    # two digits have 4 and 2 choices, so total = 4 * 2 = 8 choices
    # since we have 4 at most with N digits, so 4^N combination at most
    if digits == "":
        return []

    digit_map = {
                    "2": "abc",
                    "3": "def",
                    "4": "ghi",
                    "5": "jkl",
                    "6": "mno",
                    "7": "pqrs",
                    "8": "tuv",
                    "9": "wxyz"
                }

    ans = []
    def backtrack(start, path):

        if len(path) == len(digits):
            ans.append(''.join(path[:]))
            return
        
        # this do the trick !!
        letters = digit_map[digits[start]]
        for letter in letters:
            path.append(letter)
            backtrack(start + 1, path)
            path.pop()
    backtrack(0, [])
    return ans
        

def exist(board: list[list[str]], word: str) -> bool:
    # T: O(rows * cols * 4^len(word))
    # S: O(len(word))

    rows, cols = len(board), len(board[0])
    path = set()

    def backtrack(r, c, idx):
        if idx == len(word):
            return True
        
        if not (0 <= r < rows and 0 <= c < cols) or \
            word[idx] != board[r][c] or \
            (r, c) in path:
            return False
        
        path.add((r, c))
        ans = backtrack(r - 1, c, idx + 1) or \
                backtrack(r + 1, c, idx + 1) or \
                backtrack(r, c - 1, idx + 1) or \
                backtrack(r, c + 1, idx + 1)
        path.remove((r, c)) 
        # like path.pop(). To revoke previous choice
        return ans
    
    for r in range(rows):
        for c in range(cols):
            if backtrack(r, c, 0):
                return True
    return False

    
def solveNQueens(n: int) -> list[list[str]]:
    # T: O(n!) -> n * n-2 * n-4 ....
    # S: O(S(n)*n^2) ~ O(n^2) 
    # --> S(n) := OEIS A000170, S(9) = 352

    def create_board(state: list[list[str]]) -> list[str]:
        board = []
        for row in state:
            board.append("".join(row))
        return board
    
    # loop over row
    def backtrack(row, diags, antidiags, cols, state):
        if row == n:
            ans.append(create_board(state))
            return
        
        for col in range(n):
            cur_diag = row - col
            cur_anti_diag = row + col

            # not placeable
            if (
                col in cols
                or cur_diag in diags
                or cur_anti_diag in antidiags
            ):
                continue
            cols.add(col)
            diags.add(cur_diag)
            antidiags.add(cur_anti_diag)
            state[row][col] = "Q"
            backtrack(row+1, diags, antidiags, cols, state)
            # revoke selection
            cols.remove(col)
            diags.remove(cur_diag)
            antidiags.remove(cur_anti_diag)
            state[row][col] = "."
    
    ans = []
    init_board = [["."] * n for _ in range(n)]
    backtrack(0, set(), set(), set(), init_board)
    return ans



class Solution:
    def solveSudoku(self, board: list[list[str]]) -> None:
        # C++ would pass, while python gave TLE T.T
        # T: O(9^(n^2))
        # S: O(n^2)
        # assume that no all the init board is valid
        """
        Do not return anything, modify board in-place instead.
        """
        def is_valid(row, col, num):
            # 檢查行
            for c in range(9):
                if board[row][c] == num:
                    return False
            # 檢查列
            for r in range(9):
                if board[r][col] == num:
                    return False
            # 檢查 3x3 小方格
            start_row, start_col = 3 * (row // 3), 3 * (col // 3)
            for r in range(start_row, start_row + 3):
                for c in range(start_col, start_col + 3):
                    if board[r][c] == num:
                        return False
            return True
    
        def backtrack():
            for row in range(9):
                for col in range(9):
                    if board[row][col] == '.':
                        for num in map(str, range(1, 10)):  # 嘗試填入 1-9
                            if is_valid(row, col, num):
                                board[row][col] = num  # 暫時填入
                                if backtrack():  # 遞迴處理下一步
                                    return True
                                board[row][col] = '.'  # 回溯
                        return False  # 無法填入任何有效數字
            return True  # 全部填滿，成功解出
    
        backtrack()
    
            