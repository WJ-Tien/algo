from collections import defaultdict, deque
from typing import Optional

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

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



class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        # O(V+E) TS

        if node is None:
            return None
        
        copy_graph = dict()
        queue = deque()

        # old_node <-> new_node mapping 
        # push first node of the old_node 
        # to start the iteration
        copy_graph[node] = Node(node.val)
        queue.append(node)

        while queue:
            vertex = queue.popleft()
            for neighbor in vertex.neighbors:
                if neighbor not in copy_graph:
                    copy_graph[neighbor] = Node(neighbor.val)
                    queue.append(neighbor)
                
                # very critical, should be aware of this
                copy_graph[vertex].neighbors.append(copy_graph[neighbor])
        
        return copy_graph[node]


def countComponents(n: int, edges: list[list[int]]) -> int:
    # O (V+E) TS
    # number-of-connected-components-in-an-undirected-graph

    graph = {i: set() for i in range(n)} # O(V), O(V+E)
    visited = set() # O(V)

    for start, end in edges: # O(E)
        graph[start].add(end)
        graph[end].add(start)

    
    def dfs(vertex):
        # O (V) + O(E)
        for v in graph[vertex]: # O (V+E)
            if v not in visited:
                visited.add(v)
                dfs(v) #  O(V)
        
    ans = 0
    for vertex in graph.keys():
        if vertex not in visited:
            dfs(vertex)
            ans += 1
    return ans


def validTree(n: int, edges: list[list[int]]) -> bool:
    # graph valid tree

    if len(edges) != n - 1:
        return False

    # check cycle
    # check single component
    visited = set()
    graph = {i: set() for i in range(n)}

    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    
    def dfs(vertex, parent):
        # check cycle --> False = not valid
        visited.add(vertex)
        for neighbor in graph[vertex]:
            if neighbor in visited and neighbor != parent:
                return False
            if neighbor not in visited:
                if not dfs(neighbor, vertex):
                    return False
        return True

    return dfs(0, -1) and len(visited) == n



def findOrder(numCourses: int, prerequisites: list[list[int]]) -> list[int]:
    # O(V+E) TS

    graph = {i: set() for i in range(numCourses)}

    # 0 -> 1 - \
    #    \ -> 2 -> 3

    # 0 - \
    # 1 - 2
    # [[1,0],[1,2],[0,1]]
    # 0 -> <-1
    # 2 -> /
    in_degree = [0] * numCourses

    # topological sort
    for a, b in prerequisites:
        graph[b].add(a)
        in_degree[a] += 1
    
    queue = deque()

    for i in range(len(in_degree)):
        if in_degree[i] == 0:
            queue.append(i)

    visited = 0
    ans = []
    while queue:
        vertex = queue.popleft()
        visited += 1
        ans.append(vertex)
        for neighbor in graph[vertex]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return [] if visited < numCourses else ans 


def findMinHeightTrees(n: int, edges: list[list[int]]) -> list[int]:
    # 310. Minimum Height Trees

    # O(V+E) TS or O(V) since E = V - 1 for tree
    # remove leaves --> til remain 1 or 2 nodes --> answer
    # like peel an onion (from outermost to innermost)

    if n <= 2:
        return [i for i in range(n)]

    graph = {i: set() for i in range(n)}
    in_degree = [0] * n

    for a, b in edges:
        graph[b].add(a)
        graph[a].add(b)
        in_degree[a] += 1
        in_degree[b] += 1
    
    leaves = deque()

    for i in range(len(in_degree)):
        if in_degree[i] == 1: # leaf
            leaves.append(i)
    
    node_left = n

    # we keep the last two nodes
    # O(V+E) ~ O(V-1) ~ O(V) to do whole pruning
    # en/de queue --> O(V)
    # visit neighbor --> O(E)
    while node_left > 2:
        leaves_count = len(leaves)
        node_left -= leaves_count

        for _ in range(leaves_count):
            leaf = leaves.popleft()

            for neighbor in graph[leaf]:
                in_degree[neighbor] -= 1
                graph[neighbor].remove(leaf)

                if in_degree[neighbor] == 1:
                    leaves.append(neighbor)

    return list(leaves)

                    
def distanceK(root: TreeNode, target: TreeNode, k: int) -> list[int]:
    # 863. All Nodes Distance K in Binary Tree
    # O(N) TS

    graph = dict()
    def dfs(node, parent):
        # build graph by using dfs
        # node -> parent
        if node is None:
            return
        
        graph[node] = parent
        dfs(node.left, node)
        dfs(node.right, node)
    dfs(root, None)

    queue = deque()
    dist = 0
    queue.append((target, dist))
    ans = []
    visited = set()

    while queue:
        node, dist = queue.popleft()
        visited.add(node)

        if dist == k:
            ans.append(node.val)
            continue
        
        neighbors = [node.left, node.right, graph[node]]
        for neighbor in neighbors:
            if neighbor and neighbor not in visited:
                queue.append((neighbor, dist + 1))
    return ans



class DSU:
    def __init__(self):
        self.parent = dict()
    
    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if x != self.parent[x]: 
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        # x --> y
        self.parent[self.find(x)] = self.find(y)

class Solution:
    def accountsMerge(self, accounts: list[list[str]]) -> list[list[str]]:
        # 假設有 n 個帳戶，每個帳戶平均有 k 個電子郵件
        # per find/union operation is α(N) -> we have nk --> nka(nk)
        # Here, α(N) is the inverse Ackermann function that grows so slowly, that it doesn't exceed 4 for all reasonable N 
        # approximately N<10^600
        # a(N) = ackermann
        # NK = total emails

        # T: O(NKlogNK)
        # S: O(NK) worst case

        dsu = DSU()
        email_to_name = dict()

        for account in accounts:
            name = account[0]

            for email in account[1:]:
                email_to_name[email] = name
                dsu.union(account[1], email) 
        
        merge = dict()

        for email in email_to_name:
            root = dsu.find(email)
            if root not in merge:
                merge[root] = []
            merge[root].append(email)
        
        # {'john00@mail.com': 
        # ['johnsmith@mail.com', 'john_newyork@mail.com', 'john00@mail.com'], 
        # 'mary@mail.com': ['mary@mail.com'], 
        # 'johnnybravo@mail.com': ['johnnybravo@mail.com']}
        ans = []
        for email in merge: 
            ret = [email_to_name[email]]
            ret.extend(sorted(merge[email]))
            ans.append(ret)
        return ans
        


