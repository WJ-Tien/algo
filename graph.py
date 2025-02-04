"""
DSU: undirected
Prim: undirected (or nodes are not fully reachable)
Kruskal: undirected (otherwise it would fail to detect cycle)
Topological: DAG (directed acyclic graph)
Dijkstra/A*: non-negative weights

BFS: 1091. Shortest Path in Binary Matrix
DFS: 200. number of islands 
Dijkstra: 743. network delay time
A*: 1091. Shortest Path in Binary Matrix
Bellman-Ford: 787. Cheapest Flights Within K Stops
Kruska: 
Prim:
"""
# 弱連通weakly connected component, WCC：只要對於任意兩個點 u 和 v，至少有 "u 能走得到 v" 或 "v 能走得到 u”，就稱為弱連通。
# 強連通strongly connected component, SCC：對於任意兩個點 u 和 v，必須同時滿足 “u 能走得到 v” 以及 “v 能走得到 u”。

# Topological sort --> check dependency (e.g., num of courses)
# and check if cycles exist
# in_degree (or sometimes out_degree) == 0, put them in a deque()
# Topological Sort 是針對有向無環圖（Directed Acyclic Graph，簡稱DAG）
# 的頂點排序，使得對於每一條邊(u, v)，頂點u 都排在頂點v 的前面。 
# #這意味著，如果存在一條邊從頂點u 指向頂點v，則u 必須在v 之前出現在排序中

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
# kruscal (find MST). greedy, sort
# Prim (find MST): random choose a start, and pick min

# A* (sort of like dijkstra, but a heuristic func is needed: f(n) = g(n) + c(n))
# --> minimize f(n) (total estimate func)

# Kosaraju 算法 是另一個用來找出 強連通分量（Strongly Connected Components, SCC） 
# 的演算法。它是基於 深度優先搜尋（DFS），並且利用圖的 反向圖（transpose graph） 
# 來達成目標

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
        # S: O(n)
        # S: call stack O(n) -> O(a(n))
        self.parent = dict()
    
    # O(a(N))
    # a = inverse ackermann function
    # very small --> O(1)
    def find(self, x):
        """ return the root of x """
        # T: O(a(n)) (from O(n) -> O(a(n)))
        if x not in self.parent:
            self.parent[x] = x
        
        # path compression
        if x != self.parent[x]: 
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    # O(a(N))
    def union(self, x, y):
        # merge x & y so they become x --- root_of_x \
        #                                  y --> root_of_y
        # the concept is the root_of_x points to the root_of_y
        # T: O(a(n)) (from O(n) -> O(a(n)))
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
    # pretty much like dijkstra, except that we need a h(n)
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


def kruskal(n, edges):
    """
    Kruskal 最小生成樹演算法
    :param n: 節點數
    :param edges: [(權重, 頂點1, 頂點2), ...] 格式的邊列表
    :return: (MST 邊集合, 最小生成樹的總權重)
    """
    edges.sort()  # 按照邊的權重從小到大排序
    dsu = DSU()  # 使用提供的 DSU -> O(1) after path compression
    mst = []  # 存儲最小生成樹的邊
    total_weight = 0  # 總權重

    for weight, u, v in edges:
        if dsu.find(u) != dsu.find(v):  # 檢查是否形成環
            # find(u) == find(v)，代表 u 和 v 已經連接在同一個集合，再加入這條邊就會形成環，因此不選擇這條邊。
            dsu.union(u, v)  # 合併集合 to build MST
            mst.append((u, v, weight))  # 將邊加入 MST
            total_weight += weight  # 更新總權重

    return mst, total_weight


def prim(n, edges):
    """
    Prim's Minimum Spanning Tree (MST) Algorithm
    :param n: Number of nodes
    :param edges: List of edges [(weight, u, v)]
    :return: List of edges in the MST and the total weight of the MST
    """
    # Step 1: Create an adjacency list for the graph
    adj = {i: set() for i in range(n)}
    for weight, u, v in edges:
        adj[u].add((weight, v))
        adj[v].add((weight, u))

    # Step 2: Use a priority queue (min-heap) to store edges (weight, node)
    min_heap = []
    visited = [False] * n  # Track visited nodes
    mst_edges = []  # List to store edges in the MST
    total_weight = 0  # Variable to track the total weight of the MST

    # Start from vertex 0 (can be any vertex)
    def add_edges(node):
        visited[node] = True
        for weight, neighbor in adj[node]:
            if not visited[neighbor]:
                heapq.heappush(min_heap, (weight, neighbor, node))  # Store (weight, neighbor, node)

    # Start from vertex 0 (can be any vertex)
    add_edges(0)

    while min_heap:
        weight, node, u = heapq.heappop(min_heap)

        if visited[node]:
            continue

        # Add edge to MST (store the edge as (u, node, weight))
        mst_edges.append((u, node, weight))
        total_weight += weight
        # Add edges of the current node
        add_edges(node)

    return mst_edges, total_weight

# Test case with the same graph
edges = [
    (1, 0, 1),  # (weight 1, from vertex 0 to vertex 1)
    (3, 1, 2),  # (weight 3, from vertex 1 to vertex 2)
    (2, 0, 2),  # (weight 2, from vertex 0 to vertex 2)
    (4, 2, 3),  # (weight 4, from vertex 2 to vertex 3)
    (5, 1, 3)   # (weight 5, from vertex 1 to vertex 3)
]
n = 4  # Number of vertices
mst_edges, total_weight = prim(n, edges)
print("Minimum Spanning Tree (MST) edges: (weight, n_1, n_2)", mst_edges)
print("Total weight of MST using prim:", total_weight)


def kosaraju(graph):
    """
    使用 Kosaraju 演算法找出有向圖中所有的強連通分量 (Strongly Connected Components, SCC)
    
    :param graph: 字典表示的有向圖，格式為 {節點: [鄰居節點, ...]}
    :return: 一個列表，其中每個元素為一個強連通分量（以節點列表表示）
    """
    # 第一步：對原始圖進行一次 DFS，並在 DFS 結束後將節點推入堆疊中
    stack = []
    visited = set()

    def dfs(v):
        visited.add(v)
        for neighbor in graph.get(v, []):
            if neighbor not in visited:
                dfs(neighbor)
        stack.append(v)  # 當節點 v 的所有鄰居都探索完後，將 v 推入堆疊

    # 對圖中所有節點進行 DFS（處理可能非連通的圖）
    for v in graph:
        if v not in visited:
            dfs(v)

    # 第二步：反轉圖（將每條邊的方向反過來）
    reversed_graph = {}
    for v in graph:
        reversed_graph.setdefault(v, [])
        for neighbor in graph[v]:
            reversed_graph.setdefault(neighbor, []).append(v)

    # 第三步：以堆疊中的順序在反轉圖上進行 DFS，找出所有強連通分量
    visited.clear()  # 清空訪問紀錄，準備在反轉圖上進行新的 DFS
    sccs = []  # 用來存放所有強連通分量

    def reverse_dfs(v, component):
        visited.add(v)
        component.append(v)
        for neighbor in reversed_graph.get(v, []):
            if neighbor not in visited:
                reverse_dfs(neighbor, component)

    # 從堆疊中彈出節點，若尚未訪問則從該節點開始探索一個新的 SCC
    while stack:
        v = stack.pop()
        if v not in visited:
            component = []
            reverse_dfs(v, component)
            sccs.append(component)

    return sccs

# # 測試例子
# if __name__ == "__main__":
#     # 節點與邊的表示方法：每個節點對應一個鄰居節點的列表
#     graph = {
#         0: [1],
#         1: [2, 4, 5],
#         2: [3, 6],
#         3: [2, 7],
#         4: [0, 5],
#         5: [6],
#         6: [5],
#         7: [3, 6]
#     }
    
#     scc = kosaraju(graph)
#     print("Strongly Connected Components:", scc)
