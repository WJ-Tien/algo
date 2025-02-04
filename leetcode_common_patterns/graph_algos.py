from collections import defaultdict, deque, Counter
from heapq import heappop, heappush
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

        # path compression
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
        

def findCheapestPrice(n: int, flights: list[list[int]], src: int, dst: int, k: int) -> int:
    """

    distances[u] + w < new_distances[v] 的目的是進行
    放鬆操作」（Relaxation）
    保找到從起點經過某條邊到達目的地的更短路徑。
    如果條件成立，就更新到達節點 v 的最短距離。
    """

    prices = [float("inf")] * n
    prices[src] = 0

    for i in range(k+1): # k stops --> at most k + 1 edges
        temp_prices = prices.copy()
        for start, end, price in flights:
            if prices[start] == float("inf"):
                continue
            new_price = prices[start] + price
            if new_price < temp_prices[end]:
                temp_prices[end] = new_price

        prices = temp_prices.copy()
    
    return prices[dst] if prices[dst] != float("inf") else -1


def minKnightMoves(x: int, y: int) -> int:
    # lru_cache(maxsize=None) can replace memo by decorating dfs func

    memo = dict()

    def dfs(x, y):
        # only consider left-down or down-left
        # since these two are the minimum ways to reach (0, 0)
        x, y = abs(x), abs(y)
        if x + y == 0:
            return 0
        elif x + y == 2:
            return 2
        
        if (x, y) in memo:
            return memo[(x, y)]
        
        result = min(dfs(x-1, y-2), dfs(x-2, y-1)) + 1
        memo[(x, y)] = result
        return result
    return dfs(x, y)


def ladderLength(beginWord: str, endWord: str, wordList: list[str]) -> int:
    # O(M^2N) TS

    # for space, we have n words, each with M comb
    # and for comb, we might have M combs from other words
    # so it's N*M * M = M^2*N
    # *ot = {dot, hot, bot...}

    wordList.append(beginWord)
    # graph search
    graph = defaultdict(set)

    for word in wordList:
        for i in range(len(word)):
            pattern = word[:i] + "*" + word[i+1:]
            graph[pattern].add(word)
    
    queue = deque([(beginWord, 1)])
    visited = set(beginWord)

    while queue:
        cur_word, steps = queue.popleft()

        for i in range(len(cur_word)):
            pattern = cur_word[:i] + "*" + cur_word[i+1:]

            for neighbor in graph[pattern]:
                if neighbor == endWord:
                    return steps + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, steps+1))
    return 0


def alienOrder(words: list[str]) -> str:
    # T: O(N*M + V + E) ~ O(N*M)
    # S: O(26) 
    graph = defaultdict(set)
    in_degree = Counter()

    # kahn's algo for indegree

    for word in words:
        for char in word:
            in_degree[char] = 0
    
    for i in range(len(words)-1):
        w1, w2 = words[i], words[i+1]
        if len(w1) > len(w2) and w1[:len(w2)] == w2:
            return ""
        
        for c1, c2 in zip(w1, w2):
            if c1 != c2:
                if c2 not in graph[c1]:
                    # c1 -> c2
                    graph[c1].add(c2)
                    in_degree[c2] += 1
                break
    result = []

    # kahn's algo
    queue = deque([c for c in in_degree if in_degree[c] == 0])

    while queue:
        node = queue.popleft()
        result.append(node)
        # graph stores c1 -> c2
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    if len(result) != len(in_degree):
        return ""

    return ''.join(result)


def eventualSafeNodes(graph: list[list[int]]) -> list[int]:

    # 802. Find Eventual Safe States
    # topological sort

    # we build a rev_graph to have
    # easier access to nodes
    # u -> v ===> v -> u
    # then we collect out_degree
    # put out_degree == 0 to deque (if any)
    # as long as we reach out_degree == 0
    # put it in the queue
    # iterate over and over again
    # finally we check if the node is safe

    # O(V+E) TS

    rev_graph = [[] for _ in range(len(graph))] # easier access
    out_degree = [0] * len(graph) # ori graph
    queue = deque()

    for u in range(len(graph)):
        out_degree[u] = len(graph[u])
        for v in graph[u]:
            rev_graph[v].append(u)
    
    # 5 --> 6
    # 6 out_degree == 0
    # then 5's out_degree == 1  --> -1 --> == 0
    # 5 is safe as well
    # iterate over and over again
    for i in range(len(out_degree)):
        if out_degree[i] == 0:
            queue.append(i)
        
    safe = [False] * len(graph)
    while queue:
        # v
        node = queue.popleft()
        safe[node] = True

        for prev_node in rev_graph[node]:
            out_degree[prev_node] -= 1
            if out_degree[prev_node] == 0:
                queue.append(prev_node)

    return [i for i in range(len(graph)) if safe[i]] 


def minReorder(n: int, connections: list[list[int]]) -> int:

    # 1466. Reorder Routes to Make All Paths Lead to the City Zero
    # start from zero and think
    """

    為什麼可以確定沒有 cycle？
    在這個題目中，給定的輸入通常滿足這樣的條件：
    有 n 個城市（節點）。
    connections 中的邊數恰好為 n-1。
    從任一城市都可以通往其他城市（也就是無向圖下是連通的）。
    這正好符合樹的定義：一個連通且無循環的圖，其邊數必定為節點數 - 1。
    因此，我們可以知道底層的無向結構是一棵樹，自然不存在 cycle。

    --> what if n verticies with n edges
    --> must have a cycle

    這樣設計的直覺在於：

    !!! DFS 從 0 出發 !!!：我們希望模擬從 0 到其他城市的「反向」過程。
    當 DFS 遇到一條標記為 1 的邊時，表示這條邊的原始方向和我們需要的方向相反，必須反轉，所以累加 flip 次數。
    當 DFS 遇到一條標記為 0 的虛擬邊時，則表示該路徑已經符合讓節點最終能夠通往 0，不需要反轉。

    """

    ans = 0
    graph = {i: set() for i in range(n)}
    visited = set()

    for a, b in connections:
        graph[a].add((b, 1)) # original: need to flip
        graph[b].add((a, 0)) # artificial: no need to flip
    
    def dfs(vertex):
        nonlocal ans
        visited.add(vertex)
        for neighbor, need_flip in graph[vertex]:
            if neighbor not in visited:
                ans += need_flip
                dfs(neighbor)

    dfs(0)
    return ans


def findSmallestSetOfVertices(n: int, edges: list[list[int]]) -> list[int]:
    # 1557. Minimum Number of Vertices to Reach All Nodes
    # in-degree
    # directed-acyclic
    in_degree = [0] * n

    for u, v in edges:
        in_degree[v] += 1
    
    return [v for v in range(len(in_degree)) if in_degree[v] == 0]


def networkDelayTime(times: list[list[int]], n: int, k: int) -> int:
    # 743. Network Delay Time
    # pq stores (distance, node)
    # T:O((V+E)logV)
    # S: O(V+E)
    # Dijkstra 使用 最小堆（Min Heap），確保每次從優先隊列取出的節點 一定是當前所有未處理節點中，距離起點最短的那個。
    # 一旦某個節點被取出，它的最短距離就不會再被更改
    # 不斷從尚未確定最短路徑的節點中選擇當前距離起點最近的節點，
    # 然後更新（鬆弛）其相鄰節點的路徑長度。

    distances = {i: float("inf") for i in range(1, n+1)}
    distances[k] = 0
    pq = [(0, k)]
    visited = set()

    graph = {i: set() for i in range(1, n+1)}
    for u, v, w in times:
        graph[u].add((v, w))
    
    max_dist = 0
    
    while pq:
        cur_dist, u = heappop(pq)
        if u in visited:
            continue
            
        visited.add(u)
        max_dist = max(max_dist, cur_dist)

        for v, w in graph[u]:
            new_dist = cur_dist + w
            if new_dist < distances[v]:
                distances[v] = new_dist
                heappush(pq, (new_dist, v)) 

    return max_dist if len(visited) == n else -1



class Solution_Kruskal:
    def minCostConnectPoints(self, points: list[list[int]]) -> int:
        # T: O(N^2log(N^2)) ~ O(N^2log(N))
        # S: O(N^2)

        # build weighted graph
        # apply kruskal
        # edges = [(u, v, w)]
        edges = [] 
        for u in range(len(points)-1):
            for v in range(u+1, len(points)):
                x1 = points[u][0]
                y1 = points[u][1]
                x2 = points[v][0]
                y2 = points[v][1]
                w = self.manhattan(x1, y1, x2, y2)
                edges.append((w, u, v))

        edges.sort(key=lambda x:x[0])
        dsu = DSU()
        total_weight = 0
        for w, u, v in edges:
            if dsu.find(u) != dsu.find(v):
                dsu.union(u, v)
                total_weight += w

        return total_weight

    def manhattan(self, x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)
    

class Solution_Prim:
    def minCostConnectPoints(self, points: list[list[int]]) -> int:
        # T: O(N^2log(N^2))
        # S: O(N^2)

        # build weighted graph
        # apply Prim 
        # edges = [(u, v, w)]
        graph = {i: set() for i in range(len(points))}
        for u in range(len(points)-1):
            for v in range(u+1, len(points)):
                x1 = points[u][0]
                y1 = points[u][1]
                x2 = points[v][0]
                y2 = points[v][1]
                w = self.manhattan(x1, y1, x2, y2)
                graph[u].add((w, v))
                graph[v].add((w, u))
        pq = []
        visited = set()
        total_weight = 0

        def add_edges(node):
            visited.add(node)
            for w, v in graph[node]:
                if v not in visited:
                    heappush(pq, (w, node, v))
        
        # start from vertex 0
        add_edges(0)
        while pq:
            # E * log(E)
            # E = N^2
            # N^2log(N^2) ~ N^2log(N)
            w, u, v = heappop(pq)
            if v in visited:
                continue
            total_weight += w
            add_edges(v)
        return total_weight

    def manhattan(self, x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)