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