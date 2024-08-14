from collections import deque

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


class DFS:
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
