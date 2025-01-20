from collections import deque
from typing import List

# floodfill algorithm
# BFS or DFS
# Template

class BFS:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:

        # BFS or DFS; here BFS 
        m, n = len(image), len(image[0])
        queue = deque()
        visited = set() 
        ori_color = image[sr][sc]
        queue.append((sr, sc))
        directions = ((-1, 0), (1, 0), (0, 1), (0, -1))

        while queue: 
            r, c = queue.popleft()
            image[r][c] = color
            visited.add((r, c))
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and (nr, nc) not in visited:
                    if image[nr][nc] == ori_color:
                        queue.append((nr, nc))
        return image


class DFS_RECUR:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:

        rows, cols = len(image), len(image[0])
        ori_color = image[sr][sc]

        def dfs(r, c):
            if not (0 <= r < rows and 0 <= c < cols) \
               or image[r][c] == color \
               or image[r][c] != ori_color:
               return
            
            image[r][c] = color
            dfs(r - 1, c)
            dfs(r + 1, c)
            dfs(r, c - 1)
            dfs(r, c + 1)
        
        dfs(sr, sc)
        return image

class DFS_ITER:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:

        rows, cols = len(image), len(image[0])
        ori_color = image[sr][sc]
        stack = []
        visited = set()
        stack.append((sr, sc))

        while stack:
            r, c = stack.pop()
            image[r][c] = color
            visited.add((r, c))

            for nr, nc in [(r, c-1), (r, c+1), (r-1, c), (r+1, c)]:
                if (0 <= nr < rows and 0 <= nc < cols and image[nr][nc] == ori_color and ((nr, nc) not in visited)):
                    image[nr][nc] = color
                    stack.append((nr, nc))
        return image


def updateMatrix(mat):
    # O1 matrix
    rows, cols = len(mat), len(mat[0])
    dist = [[None] * cols for _ in range(rows)]
    queue = deque()
    seen = set()

    # Step 1: zero all zeroes 
    for r in range(rows):
        for c in range(cols):
            if mat[r][c] == 0:
                dist[r][c] = 0
                queue.append((r, c))
                seen.add((r, c))

    # Step 2: BFS
    # not visited elements are 1
    # bfs guarantees to find the shortest path
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen:
                dist[nr][nc] = dist[r][c] + 1
                queue.append((nr, nc))       
                seen.add((nr, nc))           

    return dist


def rotate_90(matrix: list[list[int]]) -> None:
    """
    rotate clockwise (90 degrees) in-place
    """

    # reverse
    matrix.reverse() # O(n), row-wise

    # transpose O(n^2)
    for i in range(len(matrix)):
        for j in range(i+1, len(matrix)):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

def rotate_180(matrix: list[list[int]]) -> None:
    """
    rotate clockwise (180 degrees) in-place
    """

    # reverse and transpose 
    matrix.reverse() # O(n), row-wise

    for row in matrix: 
        row.reverse()


def numIslands(grid: list[list[str]]) -> int:
    # O(mn) TS

    ans = 0
    visited = set()
    rows, cols = len(grid), len(grid[0])

    def dfs(r, c):
        if (not (0<=r<rows and 0<=c<cols)) or ((r, c) in visited) or (grid[r][c] == "0"):
            return

        visited.add((r, c))
        dfs(r - 1, c)
        dfs(r + 1, c)
        dfs(r, c - 1)
        dfs(r, c + 1)


    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid[r][c] == "1":
                dfs(r, c)
                ans += 1
    return ans
    

def orangesRotting(grid: list[list[int]]) -> int:

    rows, cols = len(grid), len(grid[0])
    queue = deque()
    visited = set()
    ans = 0
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    zero_counts = 0

    # multisources BFS
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c))
            if grid[r][c] == 0:
                zero_counts += 1
    
    while queue:
        n = len(queue)
        # the flag is critical here since we might revisit
        # only rotten_flag is True (fresh got changed to 2)
        # we add ans by 1
        rotten_flag = False
        for _ in range(n):
            r, c = queue.popleft()
            visited.add((r, c))

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols) and ((nr, nc) not in visited) and grid[nr][nc] == 1:
                    rotten_flag = True
                    grid[nr][nc] = 2
                    queue.append((nr, nc))
        if rotten_flag:
            ans += 1

    # important edge case
    if len(visited) + zero_counts != rows * cols:
        # 1 got isolated
        return -1
    return ans


def setZeroes(matrix: list[list[int]]) -> None:
    # T: O(mn)
    # S: O(1)
    """
    Do not return anything, modify matrix in-place instead.
    """
    rows, cols = len(matrix), len(matrix[0])
    first_row_has_zero = False
    first_col_has_zero = False

    for c in range(cols):
        if matrix[0][c] == 0:
            first_row_has_zero = True
    
    for r in range(rows):
        if matrix[r][0] == 0:
            first_col_has_zero = True
    
    # mark 1r,1c if r,c has zeroes.

    for r in range(1, rows):
        for c in range(1, cols):
            if matrix[r][c] == 0:
                matrix[0][c] = 0
                matrix[r][0] = 0

    # zero transition
    for r in range(1, rows):
        for c in range(1, cols):
            if matrix[0][c] == 0 or matrix[r][0] == 0:
                matrix[r][c] = 0
    # first row
    if first_row_has_zero:
        for c in range(cols):
            matrix[0][c] = 0
    
    if first_col_has_zero:
        for r in range(rows):
            matrix[r][0] = 0
    

def getFood(grid: List[List[str]]) -> int:
    # use traditional BFS won't work (TLE @ test case 77th)
    # mark grid[nr][nc] right after you visited
    # don't use visited set
    
    # BFS
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    # * you
    # # food
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "*":
                queue = deque()
                queue.append((r, c, 0))
                grid[r][c] = "V"
                while queue:
                    r, c, dist = queue.popleft()
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < rows and 0 <= nc < cols) and \
                            grid[nr][nc] in {"O", "#"}:
                            if grid[nr][nc] == "#":
                                return dist + 1
                            queue.append((nr, nc, dist + 1))
                            grid[nr][nc] = "V"

    return -1


def pacificAtlantic(heights: List[List[int]]) -> List[List[int]]:

    # multi-sources bfs, start from ocean

    rows, cols = len(heights), len(heights[0])

    pacific_reached = set()
    atlantic_reached = set()

    pacific_start = [(0, c) for c in range(cols)] + [(r, 0) for r in range(1, rows)]
    atlantic_start = [(r, cols-1) for r in range(rows)] + [(rows-1, c) for c in range(cols-1)] 

    def bfs(start, reached):
        queue = deque(start)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        while queue:
            r, c = queue.popleft()
            reached.add((r, c))
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols) and \
                    (nr, nc) not in reached and \
                    heights[r][c] <= heights[nr][nc]:
                    queue.append((nr, nc))
                    reached.add((r, c))
    
    bfs(pacific_start, pacific_reached)
    bfs(atlantic_start, atlantic_reached)

    return list(pacific_reached & atlantic_reached)


def longestIncreasingPath(matrix: List[List[int]]) -> int:
    # DFS + memoization
    # O(rows*cols) TS

    rows, cols = len(matrix), len(matrix[0])
    directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    memo = dict()

    def dfs(r, c):
        if (r, c) in memo:
            return memo[(r, c)]
        
        max_depth = 1

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols) and\
                (matrix[nr][nc] > matrix[r][c]):
                cur_path = 1 + dfs(nr, nc) 
                # r, c -> nr, nc
                # dfs(r, c) := max length start from r, c
                # hence if we start from r, c to nr, nc
                # the max possible length = 1 + dfs(nr, nc)
                max_depth = max(max_depth, cur_path)
        memo[(r, c)] = max_depth
        return max_depth
    
    max_depth = float("-inf")
    for r in range(rows):
        for c in range(cols):
            cur_max_depth = dfs(r, c)
            max_depth = max(max_depth, cur_max_depth)
    return max_depth


def minCost(grid: List[List[int]]) -> int:
    # 1368. Minimum Cost to Make at Least One Valid Path in a Grid
    # 0-1 BFS or dijstra (not optimal)
    
    m, n = len(grid), len(grid[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 右、左、下、上
    queue = deque([(0, 0, 0)])  # (row, col, cost)
    costs = [[float('inf')] * n for _ in range(m)]  # 儲存最小成本
    costs[0][0] = 0

    while queue:
        x, y, cost = queue.popleft()

        # 如果當前成本大於記錄的最小成本，跳過
        if cost > costs[x][y]:
            continue

        for i, (dx, dy) in enumerate(directions):
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n:
                # 計算移動成本
                new_cost = cost if grid[x][y] == i + 1 else cost + 1
                if new_cost < costs[nx][ny]:
                    costs[nx][ny] = new_cost
                    if grid[x][y] == i + 1:
                        queue.appendleft((nx, ny, new_cost))  # 成本為 0，加入佇列前端
                    else:
                        queue.append((nx, ny, new_cost))  # 成本為 1，加入佇列後端

    return costs[m-1][n-1]