"""
tree --> n verticies with n - 1 edges (connected and acyclic)
with n edges --> must be a cycle
# In a “graph with no negative-weight cycles” with N vertices, 
# the shortest path between any two vertices has at most N-1 edges.

Dijkstra: non-negative weighted graph, single source shortest path (all nodes)
A*: non-negative weighted graph, single source to single target shortest path

Topological: Directed Acyclic Graph

Bellman-Ford: directed graph (undirected non-negative weights -> dijkstra), positive/negative weighted grpah (NO negative cycle. But it can be detected), single source shortest path (to all other nodes)
SPFA (optimized bellmand-ford): directed graph (undirected non-negative weights), positive/negative weighted single source shortest path (to all other nodes)
    # Bellman-Ford 可能會對 所有節點 進行鬆弛 V−1 次。
    # SPFA 只會對「最短距離剛被更新的節點」進行處理。
    # both specialized in directed negative weight graph
Floyd-Warshall: All pairs shortest path. directed/undirected grpah, positive/negative weighted graph (not negative cycle, but can detect)

DSU: undirected
Prim: undirected (or nodes are not fully reachable)
Kruskal: undirected (otherwise it would fail to detect cycle)

Kosaraju: Directed, 強連通分量（Strongly Connected Components, SCC） (graph + rev_graph + stack0)

BFS: 1091. Shortest Path in Binary Matrix
DFS: 200. number of islands 
Dijkstra: 743. network delay time
A*: 1091. Shortest Path in Binary Matrix
Bellman-Ford: 787. Cheapest Flights Within K Stops
Kruska (DSU): 1584. Min Cost to Connect All Points
Prim (heap): 1584. Min Cost to Connect All Points
Floyd-Warshall: 1334. Find the City With the Smallest Number of Neighbors at a Threshold Distance0    
SPFA: 1334. Find the City With the Smallest Number of Neighbors at a Threshold Distance0    
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

# dijkstra: greedy-like algo, positive weighted. single source shortest path.
#           Find the shortest path from a source to another vertex. minimize g(n)
# bellman ford: postive/negative weight (sum) cyclic graph
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
# https://www.youtube.com/watch?v=Rs6DXyWpWrI&t=816s&ab_channel=Techdose

import heapq
from collections import deque

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
        elif x != self.parent[x]: 
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
    
    # we can only relax vertices from non-inf parent verticies
    # e.g. A(0) -> 100 -> B(+inf) -> B is updated as 100
    # e.g. C(+inf) -> 200 -> D(+inf) -> D remains +inf
    # for each time, we will relax the nearest one (parent != +inf)
    # for inf, we just skip it this time
    # hence if we go k times -> we determine kth layers

    # 步驟2：重複鬆弛所有邊 V-1 次 # edge relaxation --> find min distance
    for _ in range(vertices - 1):
        for u, v, w in graph:  # 每個邊 (u到v的權重為w)
            if distance[u] != float("inf") and distance[u] + w < distance[v]:
                distance[v] = distance[u] + w
    
    # 步驟3：檢查負權重循環
    # keep looping, find smaller one -> negative cycle
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

# # Test case with the same graph
# edges = [
#     (1, 0, 1),  # (weight 1, from vertex 0 to vertex 1)
#     (3, 1, 2),  # (weight 3, from vertex 1 to vertex 2)
#     (2, 0, 2),  # (weight 2, from vertex 0 to vertex 2)
#     (4, 2, 3),  # (weight 4, from vertex 2 to vertex 3)
#     (5, 1, 3)   # (weight 5, from vertex 1 to vertex 3)
# ]
# n = 4  # Number of vertices
# mst_edges, total_weight = prim(n, edges)
# print("Minimum Spanning Tree (MST) edges: (weight, n_1, n_2)", mst_edges)
# print("Total weight of MST using prim:", total_weight)


def kosaraju(graph, rev_graph):
    # O(V+E) TS

    """
    scc_1 --> scc_2 --> scc_3  stack = [3,2,1]
    scc_1 <-- scc_2 <-- scc_3  (cut property) (we don't revisit)
    
    """
    
    """
    1 -> 2 -> 3
         ^    
         |    |
              U
         4 <- 5
    """
    # 第一階段：對原圖 graph 執行 DFS，記錄每個節點的「完成順序」
    visited = set()   # 記錄已拜訪的節點
    order = []        # 用來儲存 DFS 完成時的順序

    def dfs(u):
        visited.add(u)
        for v in graph[u]:
            if v not in visited:
                dfs(v)
        order.append(u)  # 當 u 的所有鄰接點都拜訪完畢後，再加入 order

    # 對圖中所有節點進行 DFS（確保沒有遺漏孤立的節點）
    for node in graph:
        if node not in visited:
            dfs(node)
    # [5, 4, 3, 2, 1]
    # 第二階段：對反向圖 rev_graph 執行 DFS，
    # 依據在第一階段所得的 order（反向順序）找出各個強連通分量 (SCC)
    visited.clear()   # 清空 visited 以便重複使用
    scc_list = []     # 用來儲存所有的強連通分量

    # using CUT property here
    """
    1 <- 2 <- 3
              ^
         |    |
         U    
         4 -> 5
    """
    def reverse_dfs(u, component):
        visited.add(u)
        component.append(u)
        for v in rev_graph[u]:
            if v not in visited:
                reverse_dfs(v, component)

    # 依照第一階段 DFS 完成順序的反向順序進行第二次 DFS
    # [1, 2, 3, 4, 5]
    for node in reversed(order):
        if node not in visited:
            component = []
            reverse_dfs(node, component)
            scc_list.append(component)

    return scc_list


# # 測試範例
# if __name__ == "__main__":
#     # 定義原圖 (graph) 與反向圖 (rev_graph)
#     graph = {
#         1: [2],
#         2: [3],
#         3: [1, 4],
#         4: [5],
#         5: [6],
#         6: [4, 7],
#         7: []
#     }

#     # 反向圖中，每條邊的方向與原圖相反
#     rev_graph = {
#         1: [3],
#         2: [1],
#         3: [2],
#         4: [3, 6],
#         5: [4],
#         6: [5],
#         7: [6]
#     }

#     scc = kosaraju(graph, rev_graph)
#     print("Strongly Connected Components (強連通分量):", scc)

def floyd_warshall(graph):
    # T: O(n^3)
    # S: O(n^2)
    # n 通常指的是「圖中節點（頂點）的數量」。
    # 我們假設要「嘗試把節點 k 當作中繼點」，意即看能不能透過節點 k，讓 i 到 j 的路徑縮短。
    # dist[i][i] <0 --> negative cycle
    """
    計算圖中所有節點對之間的最短路徑。
    
    參數:
        graph: 二維列表（矩陣），其中 graph[i][j] 表示從節點 i 到節點 j 的邊權重。
               如果 i 與 j 之間沒有直接邊，請將對應值設定為 float('inf')（無限大）。
               
    回傳:
        dist: 二維列表，dist[i][j] 為從節點 i 到節點 j 的最短距離。
              若存在負權環，則演算法可能無法正確處理（此實作假設圖中沒有負環）。
    """
    n = len(graph)  # 節點數
    # 建立距離矩陣，初始時直接複製原圖的權重矩陣
    dist = [row[:] for row in graph]

    # 三層迴圈依序嘗試每個中繼點 k
    for k in range(n):
        # 對每一對起點 i 與終點 j
        for i in range(n):
            for j in range(n):
                # 如果經過中繼點 k 的路徑比原本 i -> j 的路徑更短，就更新距離
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist

# 測試範例
# if __name__ == "__main__":
#     # 定義無向圖或有向圖的權重矩陣
#     # 節點間沒有直接相連的邊，則用無限大（INF）表示
#     INF = float('inf')
#     graph = [
#         [0,   3,   INF, 7],
#         [8,   0,   2,   INF],
#         [5,   INF, 0,   1],
#         [2,   INF, INF, 0]
#     ]

#     # 呼叫 Floyd-Warshall 演算法取得所有節點對之最短距離
#     shortest_paths = floyd_warshall(graph)

#     # 印出結果
#     print("所有節點對的最短路徑距離矩陣：")
#     for row in shortest_paths:
#         print(row)


def spfa(n, edges, source):
    # optimized bellman-ford
    # Shortest Path Faster Algorithm
    """
    使用 SPFA 演算法計算從單一源點 (source) 到其他所有頂點的最短路徑距離。
    
    參數：
    n      : 圖中頂點的數目（頂點編號預設為 0 ~ n-1）。
    edges  : 鄰接表形式的邊資訊，edges[u] = [(v1, w1), (v2, w2), ...]
             表示從節點 u 可以走到節點 v1, v2, ...，權重(或距離) 分別為 w1, w2, ...
    source : 單一源點的編號。
    
    回傳：
    dist   : 一維列表，dist[v] 表示從 source 到頂點 v 的最短距離。
             若無法到達，則會保持為 float('inf')。
    
    時間複雜度：在最差情況下依舊可能達到 O(n * m)，
                其中 n 為頂點數、m 為邊數。
                但在許多情況下，SPFA 通常比 Bellman-Ford 實際執行更快。
    """
    INF = float('inf')
    
    # dist[v] = 目前從 source 到 v 的最短距離，初始全部設為 INF
    dist = [INF] * n
    dist[source] = 0  # 自己到自己距離為 0
    
    # in_queue[v] 用來標記節點 v 是否已經在佇列中
    in_queue = [False] * n
    
    # 使用雙向佇列 deque 實作 FIFO
    queue = deque()
    queue.append(source)
    in_queue[source] = True
    
    # 當佇列不空時，就持續進行鬆弛操作
    while queue:
        u = queue.popleft()
        in_queue[u] = False  # 把 u 移出佇列
        
        # 針對所有從 u 出發可到達的 (v, w)，嘗試更新 dist[v]
        for v, w in edges[u]:
            # 如果透過 u 能讓 dist[v] 更小，就更新
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                # 若 v 還不在佇列中，才將其加入，避免重複
                if not in_queue[v]:
                    queue.append(v)
                    in_queue[v] = True
    
    return dist

# # 測試示例
# if __name__ == "__main__":
#     """
#     這裡示範一個有 5 個節點 (0, 1, 2, 3, 4) 的有向圖。
#     edges[u] = [(v, w), ...]：
#       - 從 u -> v 的權重為 w
#     """
#     n = 5  # 頂點數量
#     edges = [[] for _ in range(n)]
    
#     # 建立邊 (範例)
#     # 0 -> 1 (權重 2)
#     edges[0].append((1, 2))
#     # 0 -> 2 (權重 5)
#     edges[0].append((2, 5))
#     # 1 -> 2 (權重 1)
#     edges[1].append((2, 1))
#     # 1 -> 3 (權重 3)
#     edges[1].append((3, 3))
#     # 2 -> 4 (權重 2)
#     edges[2].append((4, 2))
#     # 3 -> 4 (權重 4)
#     edges[3].append((4, 4))
    
#     source = 0
#     dist = spfa(n, edges, source)
    
#     print(f"從節點 {source} 出發到各頂點的最短距離：")
#     for v in range(n):
#         print(f"dist[{v}] = {dist[v]}")
