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