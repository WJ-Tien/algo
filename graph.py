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


# dijkstra: greedy-like algo, unweighted graph. single source shortest path.
#           Find the shortest path from a source to another vertex
# bellman ford: acyclic & postive weight (sum) cyclic graph
#           single source shortest path

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