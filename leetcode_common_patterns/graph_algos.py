from collections import defaultdict, deque

# Normal DFS
def canFinish(numCourses: int, prerequisites: list[list[int]]) -> bool:
    """
    時間複雜度（Time Complexity, TC） 
    在這道題目中，我們用深度優先搜尋（DFS）遍歷圖，並檢查是否有環存在。 
    1. 構建圖的時間 graph = {i: [] for i in range(numCourses)} 初始化空圖的時間複雜度為 𝑂 ( 𝑉 ) O(V)，其中 𝑉 V 是節點數，也就是課程數 numCourses。 
    for a, i in prerequisites: graph[a].append(i) 遍歷 prerequisites 並將邊加入圖，時間複雜度為 𝑂 ( 𝐸 ) O(E)，其中 𝐸 E 是先修條件數，也就是邊的數量。 因此，構建圖的時間複雜度為： 𝑂 ( 𝑉 + 𝐸 ) 
    O(V+E) 
    2. DFS 遍歷圖的時間 在最壞情況下，每個節點會被訪問一次，並且每條邊也只會被檢查一次。 訪問所有節點的時間複雜度為 𝑂 ( 𝑉 ) O(V)。 
    檢查所有邊的時間複雜度為 𝑂 ( 𝐸 ) O(E)。 因此，DFS 的時間複雜度為： 𝑂 ( 𝑉 + 𝐸 ) 
    O(V+E) 總時間複雜度 圖的構建和 DFS 遍歷是順序執行的，
    總的時間複雜度為： 𝑂 ( 𝑉 + 𝐸 ) O(V+E) 
    空間複雜度（Space Complexity, SC） 
    1. 圖的存儲 用一個字典 graph 存儲有向圖，字典中每個節點存的是一個鄰接列表。 
       列表的大小總計為 𝑂 ( 𝐸 ) O(E)。 
    圖結構的空間複雜度為： 𝑂 ( 𝑉 + 𝐸 ) O(V+E) 
    2. 遞歸棧的深度 使用 DFS 時，遞歸棧的最大深度取決於圖的深度，最壞情況下深度可以是 𝑂 ( 𝑉 ) O(V)。 
    這部分的空間複雜度為： 𝑂 ( 𝑉 ) O(V) 
    3. 狀態表的存儲 使用 status 字典來記錄每個節點的訪問狀態。 
    該字典的大小為 𝑂 ( 𝑉 ) O(V)。 
    總空間複雜度 所有空間需求相加，總空間複雜度為： 𝑂 ( 𝑉 + 𝐸 ) O(V+E) 
    總結 時間複雜度 𝑇 𝐶 TC： 𝑂 ( 𝑉 + 𝐸 ) 
    O(V+E) 空間複雜度 𝑆 𝐶 SC： 𝑂 ( 𝑉 + 𝐸 ) O(V+E)

    """
    # O(V+E) TS
    graph = {i: set() for i in range(numCourses)} #T: O(V)
    status = {i: 0 for i in range(numCourses)} #T: O(V)

    for a, b in prerequisites:
        graph[a].add(b) #T: O(E)

    # 0: not visited
    # 1: finished (checked OK)
    # -1: checking

    # Return true if cyclic
    # meet at vertex w/ -1 twice or more.
    def dfs(vertex):
        #T: O(V+E)
        #S: O(V)

        if status[vertex] == 1:
            return False
        elif status[vertex] == -1:
            return True
        
        # status 0 --> -1
        status[vertex] = -1 

        for v in graph[vertex]:
            if dfs(v):
                return True

        status[vertex] = 1
        return False
    
    for c in list(graph.keys()):
        if dfs(c):
            return False
    return True


# using topological sort
# in_degree = 0 will be placed into the queue (deque)
# lower in_degree --> higher in_degree ("sort")
# if cyclic (no in_degree == 0 vertices), you cannot use topo sort (course schedule)
# since we cannot proceed by push in_degree == 0 to the queue
def canFinish_topo_sort(numCourses: int, prerequisites: list[list[int]]) -> bool:

    graph = defaultdict(set)
    in_degree = [0] * numCourses

    for a, b in prerequisites:
        # for in_degree, we take b as key
        # b -> a 
        graph[b].add(a) # adjacent set
        in_degree[a] += 1
    
    queue = deque()
    for i in range(len(in_degree)):
        if in_degree[i] == 0:
            queue.append(i)
    
    visited = 0
    while queue:
        vertex = queue.popleft()
        visited += 1 # for in_degree = 0's node. check OK so add 1

        for neighbor in graph[vertex]:
            in_degree[neighbor] -= 1 # consume neighbor's in_degree !!
            if in_degree[neighbor] == 0: # Only add in_degree == 0 to the queue
                queue.append(neighbor)

    return visited == numCourses


