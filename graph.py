"""
BFS: 1091. Shortest Path in Binary Matrix
DFS: 200. number of islands 
Dijkstra: 743. network delay time
A*: 1091. Shortest Path in Binary Matrix
Bellman-Ford: 787. Cheapest Flights Within K Stops
Kruska: 
Prim:
"""
# Graph Theory
# topological sort --> check dependency (e.g., num of courses)
# and check if cycles exist
# in_degree (or sometimes out_degree) == 0, put them in a deque()

# BFS: start from a source and to another vertex, it will find the shortest path in between
# but the it should be applied to an unweighted graph.

# dijkstra: greedy-like algo, positive weighted/cyclic graph. single source shortest path.
#           Find the shortest path from a source to another vertex. minimize g(n)
# bellman ford: acyclic & postive/negative weight (sum) cyclic graph
#           single source shortest path

# A spanning tree is a connected subgraph in an undirected graph 
# where all vertices are connected with the minimum number of edges
# 1. 一個圖(Graph)不只會有一個生成樹(Spanning Tree)
# 2. 同一個圖的生成樹，會有相同的點 (Vertex), 邊(Edge)的個數。
# 3. 所有的生成數，不會出現 Cycle, loop的結構。
# 4. 如果把其中一個「邊(Edge)」拿掉就會成為無連通圖(Disconnected Graph)我們可以說生成樹是最小的連接(minimally connected)
# 5. 只要多加一個邊，就會形成cycle, Loop，所以也可以成生成樹是maximally acyclic(最大無環)
# A - B
# |   |
# C   D
# MST: spanning tree that gives the min edges weight sum
# https://mycollegenotebook.medium.com/%E7%94%9F%E6%88%90%E6%A8%B9-spanning-tree-fa19df652efb

# https://ithelp.ithome.com.tw/articles/10277930
# kruscal (find MST). sort
# Prim (find MST): random choose a start, and pick min

# A* (sort of like dijkstra, but a heuristic func is needed: f(n) = g(n) + c(n))
# --> minimize f(n) (total estimate func)

import heapq

def has_cycle_undirected(graph):
    """
    graph: dict[node, set[node]]
        An adjacency representation of an undirected graph.
        Example:
            {
                1: {2, 3},
                2: {1, 4},
                3: {1},
                4: {2}
            }
    Returns:
        True if the undirected graph contains at least one cycle, False otherwise.
    """
    visited = set()
    
    def dfs(current, parent):
        visited.add(current)
        
        for neighbor in graph[current]:
            # Case 1: Found an already visited vertex that isn't the parent
            # This means we found a cycle
            if neighbor in visited and neighbor != parent:
                return True
                
            # Case 2: Unvisited vertex - explore it using DFS    
            if neighbor not in visited:
                if dfs(neighbor, current):
                    return True
                    
        return False
    
    for vertex in graph:
        if vertex not in visited:
            if dfs(vertex, None):
                return True
                
    return False


class DSU:
    def __init__(self):
        self.parent = dict()
    
    # O(a(N))
    # a = ackermann function
    # very small --> O(1)
    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if x != self.parent[x]: 
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    # O(a(N))
    def union(self, x, y):
        # x --> y
        self.parent[self.find(x)] = self.find(y)



def bellman_ford(graph, vertices, source):
    # T: O(EV)
    # S: O(V)
    # 步驟1：初始化距離
    distance = [float("inf")] * vertices
    distance[source] = 0
    
    # 步驟2：重複鬆弛所有邊 V-1 次 # edge relaxation --> find min distance
    for _ in range(vertices - 1):
        for u, v, w in graph:  # 每個邊 (u到v的權重為w)
            if distance[u] != float("inf") and distance[u] + w < distance[v]:
                distance[v] = distance[u] + w
    
    # 步驟3：檢查負權重循環
    for u, v, w in graph:
        if distance[u] != float("inf") and distance[u] + w < distance[v]:
            return False  # 存在負權重循環
    
    return distance

def dijkstra(graph, start):
    # T: O((V+E)logV)
    # S: O(V+E)
    # Initialize distances with infinity
    # Dijkstra 使用 最小堆（Min Heap），確保每次從優先隊列取出的節點 一定是當前所有未處理節點中，距離起點最短的那個。
    # 一旦某個節點被取出，它的最短距離就不會再被更改
    # 不斷從尚未確定最短路徑的節點中選擇當前距離起點最近的節點，
    # 然後更新（鬆弛）其相鄰節點的路徑長度。
    distances = {node: float('inf') for node in graph}
    distances[start] = 0  # Distance to the start node is 0
    priority_queue = [(0, start)]  # (distance, node)
    visited = set()

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_node in visited:
            continue
        visited.add(current_node)

        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances


# A* algorithm
def manhattan_heuristic(x, y, goal_x, goal_y):
    return abs(goal_x - x) + abs(goal_y - y)

def astar_manhattan(grid, start, goal):
    # T: best O(E), average/worst O(E+V)logV
    # S: O(V+E)
    # g(n) = number of steps taken so far
    # h(n) = Manhattan Distance heuristic (estimate func)
    # f(n) = g(n) + h(n). Total estimate cost function
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

    start_x, start_y = start
    goal_x, goal_y = goal
    
    pq = [(0 + manhattan_heuristic(start_x, start_y, goal_x, goal_y), 0, start_x, start_y)]
    # (f(n), g(n), x, y)
    visited = set()
    g_costs = { (start_x, start_y): 0 }  # Stores the best cost found so far

    while pq:
        _, g, x, y = heapq.heappop(pq)

        if (x, y) == goal:
            return g  # Found shortest path

        if (x, y) in visited:
            continue
        visited.add((x, y))

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
                new_g = g + 1
                if (nx, ny) not in g_costs or new_g < g_costs[(nx, ny)]:
                    g_costs[(nx, ny)] = new_g
                    f_cost = new_g + manhattan_heuristic(nx, ny, goal_x, goal_y)
                    heapq.heappush(pq, (f_cost, new_g, nx, ny))
                    # dijkstra only consider g(n)
                    # A* consider f(n)

    return -1  # No path found
