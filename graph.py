
# topological sort --> check dependency (e.g., num of courses)
# and check if cycles exist
# in_degree (or sometimes out_degree) == 0, put them in a deque()

# BFS: start from a source and to another vertex, it will find the shortest path in between
# but the it should be applied to an unweighted graph.

# dijkstra: greedy-like algo, weighted graph. single source shortest path.
#           Find the shortest path from a source to another vertex
# bellman ford: acyclic & postive weight (sum) cyclic graph
#           single source shortest path

# TODO
# A*
# kruscal
# Prim

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
