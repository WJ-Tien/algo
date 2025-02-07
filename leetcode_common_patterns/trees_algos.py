
from collections import deque
from typing import Optional

"""
陣列索引

實際用於儲存節點。
左子節點是 2 * i + 1。
右子節點是 2 * i + 2。
位置標記

用於模擬節點在二叉樹層次中的位置。
左子節點是 2 * 當前位置。
右子節點是 2 * 當前位置 + 1。

"""

# tree template

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def maxDepth_iter(root: Optional[TreeNode]) -> int:
    if root is None:
        return 0
    
    stack = [(root, 1)]
    max_depth = 0

    while stack:
        node, depth = stack.pop()
        max_depth = max(max_depth, depth) 

        if node.left:
            stack.append((node.left, depth + 1))
        if node.right:
            stack.append((node.right, depth + 1))

    return max_depth


def maxDepth_recur(root: Optional[TreeNode]) -> int:

    if root is None: 
        return 0
    
    return 1 + max(maxDepth_recur(root.left), maxDepth_recur(root.right))


def isBalanced_recur(root: Optional[TreeNode]) -> bool:
    
    def dfs(root):
        if root is None:
            return (True, 0) # (isBalanced, cur_max_height)
        
        lh = dfs(root.left)
        rh = dfs(root.right)

        # isbalance of the individual tree does not always guarantee  \
        # to create balanced subtree
        # isbalanced = True for left/right substree, but their height diff may larger than one
        is_balanced = (lh[0] and rh[0]) and \
                        (abs(lh[1] - rh[1]) <= 1)
        return (is_balanced, 1 + max(lh[1], rh[1]))
    return dfs(root)[0]


def isBalanced_iter(root: Optional[TreeNode]) -> bool:
        if not root:
            return True

        stack = deque()
        depth = {}

        # state := False --> not yet handled 
        stack.append((root, False))

        while stack:
            node, visited = stack.pop()

            if node is None:
                continue

            if not visited:
                stack.append((node, True)) # key !
                stack.append((node.right, False))
                stack.append((node.left, False))
            else:
                # when a node is visited twice
                # --> cal height
                left_depth = depth.get(node.left, 0)
                right_depth = depth.get(node.right, 0)

                if abs(left_depth - right_depth) > 1:
                    return False

                depth[node] = 1 + max(left_depth, right_depth)

        return True


def diameterOfBinaryTree(root: Optional[TreeNode]) -> int:
    """
    diameter = lh + rh 
    lh, rh := max_depth(max_height) of the left and right subtree respectively
    1 + max(lh, rh) := max_depth(max_height) of the node ==
                    := return of dfs func
    """

    ans = float("-inf")

    def dfs(root):
        nonlocal ans
        if root is None:
            return 0
        
        lh = dfs(root.left)
        rh = dfs(root.right)
        d = lh + rh
        ans = max(ans, d)

        return 1 + max(lh, rh)
    dfs(root) 
    return ans


def isSameTree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:

    if p is None and q is None:
        return True
    elif p is None or q is None: 
        return False
    else:
        if p.val == q.val:
            return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)
        return False


def isSymmetric(root: Optional[TreeNode]) -> bool:

    # using sametree concept
    # dfs is better, bfs is way too complicated

    def dfs(p, q):
        if p is None and q is None:
            return True
        elif p is None or q is None:
            return False
        else:
            if p.val == q.val:
                return dfs(p.left, q.right) and dfs(p.right, q.left)
            return False
    return dfs(root.left, root.right)


def sortedArrayToBST(nums: list[int]) -> Optional[TreeNode]:

    # in-order placement
    def dfs(start, end):
        if start > end:
            # reach leaf
            return None
        mid = start + (end - start) // 2
        root = TreeNode(nums[mid])

        root.left = dfs(start, mid - 1)
        root.right = dfs(mid + 1, end)
        return root
    
    return dfs(0, len(nums) - 1)

def lowestCommonAncestor_BST_ONLY(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':

    cur = root

    while cur:
        if cur.val > p.val and cur.val > q.val:
            cur = cur.left
        elif cur.val < p.val and cur.val < q.val:
            cur = cur.right
        else:
            return cur

def lowestCommonAncestor(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':

    # 236. Lowest Common Ancestor of a Binary Tree
    # general version of binary tree (or BST)
    # O(N) TS

    parent = dict() # cur_node to parent mapping
    queue = deque()
    queue.append(root)
    parent[root] = None # root has no parent

    while queue:
        node = queue.popleft()

        if node.left:
            parent[node.left] = node
            queue.append(node.left)
        if node.right:
            parent[node.right] = node
            queue.append(node.right)

    p_ancestors = set()
    while p:
        p_ancestors.add(p)
        p = parent[p]
    
    # bottom to top
    while q:
        if q in p_ancestors:
            return q
        q = parent[q]

    return None


def levelOrder(root: Optional[TreeNode]) -> list[list[int]]:

    if root is None:
        return []

    queue = deque()
    queue.append(root)
    ans = []

    while queue:
        n = len(queue)
        level = []
        for _ in range(n):
            node = queue.popleft()
            level.append(node.val)
    
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        ans.append(level)
    return ans


def hasPathSum(root: Optional[TreeNode], targetSum: int) -> bool:
    # TC: O(N), O(N)
    # SC: O(logN), O(N)

    def dfs(root, cur_sum):
        if root is None:
            return False
        
        cur_sum += root.val
        if cur_sum == targetSum and root.left is None and root.right is None:
            # early termination, since we only need to know if any ONE is true
            return True
        return dfs(root.left, cur_sum) or dfs(root.right, cur_sum)

    return dfs(root, 0)


def pathSumII(root: Optional[TreeNode], targetSum: int) -> list[list[int]]:
    # average O(N*logN) TS
    # worst/skewed O(N^2) TS

    ans = []
    def dfs(root, cur_sum, path):
        if root is None:
            return
        
        cur_sum += root.val
        path.append(root.val)
        if cur_sum == targetSum and root.left is None and root.right is None:
            # can't early terminate here, we have to collect all
            ans.append(path[:])
            
        dfs(root.left, cur_sum, path)
        dfs(root.right, cur_sum, path)
        path.pop() # very important, inspired by backtracking
        
    dfs(root, 0, [])
    return ans
    

def isValidBST(root: Optional[TreeNode]) -> bool:
    # O(N) T
    # O(H) S

    low, high = float("-inf"), float("inf")

    def dfs(root, low, high):
        # leaves
        if root is None:
            return True
        
        # very important here, we must loop until the leaves
        # so we cannot stop early
        if not (low < root.val < high):
            return False
        return dfs(root.left, low, root.val) and dfs(root.right, root.val, high)

    return dfs(root, low, high)


def buildTree(preorder: list[int], inorder: list[int]) -> Optional[TreeNode]:
    # 105. Construct Binary Tree from Preorder and Inorder Traversal
    # O(n) TS

    # 前序遍歷告訴我們根節點在哪裡 [0] = root
    # 中序遍歷告訴我們如何分割左右子樹 [0] = left_node
    # 通過計算左子樹的大小，我們能在前序遍歷中找到右子樹的起始位置
    # 讓我們分析這個公式：pre_start + (root_index - in_start + 1)

    # pre = [根節點, 左子樹的所有節點..., 右子樹的所有節點...]
    # in  =[左子樹的所有節點...,根節點..., 右子樹的所有節點...] 
    # root_index - in_start 這部分計算的是左子樹的節點數量
    # 例如：
    # 中序遍歷：[9, 3, 15, 20, 7]
    #         ^  ^
    #         |  |
    #     in_start root_index
    
    # root_index(1) - in_start(0) = 1
    # 確實，3 左邊只有一個節點(9)
    # 為什麼要 +1？
    # 因為還要跳過根節點本身
    # 所以是：左子樹的節點數 + 1個根節點
    
    def helper(pre_start, in_start, in_end):
        if pre_start >= len(preorder) or in_start > in_end:
            return None
    
        root_val = preorder[pre_start]
        root = TreeNode(root_val)
    
        root_index = inorder_index_map[root_val] # == mid
    
        # Left subtree: preorder index + 1, inorder range [in_start, root_index - 1]
        root.left = helper(pre_start + 1, in_start, root_index - 1)
    
        # Right subtree: preorder index + (root_index - in_start + 1), inorder range [root_index + 1, in_end]
        root.right = helper(pre_start + (root_index - in_start + 1), root_index + 1, in_end)
    
        return root

    # Create a map for quick lookup of root indices in the inorder list.
    inorder_index_map = {val: idx for idx, val in enumerate(inorder)}
    return helper(0, 0, len(inorder) - 1)


def buildTree_opt(preorder, inorder):
    # 建立字典(哈希表)，記錄inorder中每個值對應的索引位置，方便O(1)查找
    in_map = {}
    for i, val in enumerate(inorder):
        in_map[val] = i
    
    # 定義遞迴函式
    def helper(pre_left, pre_right, in_left, in_right):
        # 若區間無效，表示子樹為空，返回 None
        if pre_left > pre_right or in_left > in_right:
            return None
        
        # 前序遍歷的第一個位置就是root
        root_val = preorder[pre_left]
        root = TreeNode(root_val)
        
        # 在inorder中找到root的位置
        in_root_index = in_map[root_val]
        
        # 左子樹的大小 (根在inorder中的索引 - 當前inorder的起始位置)
        left_size = in_root_index - in_left
        
        # 建立左子樹
        root.left = helper(
            pre_left + 1,               # 左子樹在preorder的起點 (root往後一格)
            pre_left + left_size,       # 左子樹在preorder的終點
            in_left,                    # 左子樹在inorder的起始
            in_root_index - 1           # 左子樹在inorder的終點 (根位置往前一格)
        )
        
        # 建立右子樹
        root.right = helper(
            pre_left + left_size + 1,   # 右子樹在preorder的起點
            pre_right,                  # 右子樹在preorder的終點
            in_root_index + 1,          # 右子樹在inorder的起點 (根位置往後一格)
            in_right                    # 右子樹在inorder的終點
        )
        
        return root
    
    # 呼叫遞迴函式，索引涵蓋整個序列範圍
    return helper(0, len(preorder) - 1, 0, len(inorder) - 1)


def widthOfBinaryTree(root: Optional[TreeNode]) -> int:

    # O(n) TS
    # 662. Maximum Width of Binary Tree

    queue = deque()
    queue.append((root, 0)) # node, pos
    ans = 0

    while queue:
        n = len(queue)
        _, level_start = queue[0]

        for _ in range(n):
            node, pos = queue.popleft()

            if node.left:
                queue.append((node.left, 2*pos))
            if node.right:
                queue.append((node.right, 2*pos + 1))

        ans = max(ans, pos - level_start + 1)

    return ans 


def inorderSuccessor(root: TreeNode, p: TreeNode) -> Optional[TreeNode]:

    found = False
    successor = None

    def dfs(root):
        nonlocal found, successor
        # root is None  or already found successor
        if not root or successor:
            return
        
        dfs(root.left)

        if found and not successor:
            successor = root
            return

        if root == p: 
            found = True
        
        dfs(root.right)
    dfs(root)
    return successor


def pathSum(root: Optional[TreeNode], targetSum: int) -> int:

    ans = 0
    prefix_sum = {0: 1} 

    def dfs(root, cur_sum):
        nonlocal ans
        if root is None:
            return
        
        cur_sum += root.val
        if cur_sum - targetSum in prefix_sum:
            ans += prefix_sum[cur_sum - targetSum]
        
        prefix_sum[cur_sum] = prefix_sum.get(cur_sum, 0) + 1

        dfs(root.left, cur_sum)
        dfs(root.right, cur_sum)
        prefix_sum[cur_sum] -= 1 # backtrack; to retract decision
    
    dfs(root, 0)
    return ans


"""
 -10
 /  \
9    20
    /  \
   15   7

1. 通過該節點計算完整路徑和
定義：完整路徑和是以當前節點為「樞紐」的總路徑，可以包含左右子樹的路徑和。
應用：用於更新全域變數 max_sum，記錄整棵樹中可能的最大路徑和。
例如：在節點 20，完整路徑和是 15 → 20 → 7，這條路徑的和為 20 + 15 + 7 = 42。
2. 回傳父節點的貢獻值
定義：當節點作為其父節點路徑的一部分時，只能選擇「左子樹」或「右子樹」之一（加上自身節點值），因為路徑不能分叉。
應用：用於讓父節點繼續計算其完整路徑和。
例如：在節點 20，回傳值是 20 + max(15, 7) = 35，表示節點 20 只能選擇一個方向，然後加上自身，提供給父節點使用。
"""

def maxPathSum(root: Optional[TreeNode]) -> int:
    # 124. Binary Tree Maximum Path Sum
    max_sum = float("-inf")

    def dfs(root):
        nonlocal max_sum
        if not root:
            return 0
        
        left_max = max(dfs(root.left), 0)
        right_max = max(dfs(root.right), 0)

        max_sum = max(max_sum, root.val + left_max + right_max)

        return root.val + max(left_max, right_max)
    
    dfs(root)
    return max_sum

        
def countUnivalSubtrees(root: Optional[TreeNode]) -> int:

    # bottom up dfs
    # check left all uni and right all uni
    # and compare value

    ans = 0
    def dfs(root):
        nonlocal ans
        if root is None:
            return True

        is_left = dfs(root.left)
        is_right = dfs(root.right)

        if is_left and is_right:
            # check if false is simpler
            # negative list
            # not exist or val is the same --> True
            if root.left and root.val != root.left.val:
                return False
            if root.right and root.val != root.right.val:
                return False
            ans += 1
            return True
        return False

    dfs(root)
    return ans


def minDepth(root: Optional[TreeNode]) -> int:
    # 111. Minimum Depth of Binary Tree

    def dfs(root):
        if root is None:
            return 0
        
        ld = dfs(root.left)
        rd = dfs(root.right)

        # two edge cases, single-sided tree
        """
        Example:
         5
          \
           3
            \
             2
        """
        if root.left and not root.right:
            return 1 + ld
        if not root.left and root.right:
            return 1 + rd
        return 1 + min(ld, rd)
    
    return dfs(root)


def maxAncestorDiff(root: Optional[TreeNode]) -> int:

    ans = float("-inf")

    def dfs(root):
        nonlocal ans
        if root is None:
            # min, max
            return (float("-inf"), float("inf"))

        left_min, left_max = dfs(root.left)
        right_min, right_max = dfs(root.right)

        if left_min == float("-inf") and left_max == float("inf"):
            left_min = left_max = root.val
        if right_min == float("-inf") and right_max == float("inf"):
            right_min = right_max = root.val

        new_min = min(root.val, left_min, right_min)
        new_max = max(root.val, left_max, right_max)
        ans = max(ans, abs(root.val - new_min), abs(root.val - new_max))

        return new_min, new_max 
    
    dfs(root)
    return ans

def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
    """
    節點沒有子節點（葉子節點）： 直接刪除該節點。
    節點有一個子節點： 用其子節點取代該節點。
    節點有兩個子節點： 找到右子樹中最小值節點（或左子樹中最大值節點），將該節點值替換到目標節點，然後刪除替換節點。
    """
    # T: O(H1+H2)
    # S: O(H)
    if not root:
        return root
    
    if key < root.val:
        root.left = self.deleteNode(root.left, key)
    elif key > root.val:
        root.right = self.deleteNode(root.right, key)
    else:
        # found the position
        # case 1
        if not root.left and not root.right:
            return None
        
        # case 2
        if not root.left and root.right:
            return root.right
        if root.left and not root.right:
            return root.left
        
        # case 3
        # either left or right, > 1 solutions 
        # here we choose leftmost in right subtree
        min_node = self.find_min(root.right)
        root.val = min_node.val
        root.right = self.deleteNode(root.right, min_node.val)
    return root

def find_min(self, node):
    while node.left:
        node = node.left
    return node


def n_ary_maxDepth(root: TreeNode) -> int:

    def dfs(root):
        if root is None:
            return 0

        # remember this should be local var
        # instead of global var
        max_c_depth = 0
        for c in root.children:
            max_c_depth = max(max_c_depth, dfs(c)) 
        return 1 + max_c_depth

    return dfs(root)


def n_ary_postorder(root: TreeNode) -> list[int]:

    ans = []
    def dfs(root):
        if root is None:
            return 
        
        for c in root.children:
            dfs(c)
        ans.append(root.val)
        return
    
    dfs(root)
    return ans


def preorder(root: TreeNode) -> list[int]:

    ans = []

    def dfs(root):
        if root is None:
            return
        
        ans.append(root.val)
        for c in root.children:
            dfs(c)
        
        return

    dfs(root)
    return ans


def removeLeafNodes(root: Optional[TreeNode], target: int) -> Optional[TreeNode]:

    # post order traversal

    def dfs(root):
        if root is None:
            return
        
        root.left = dfs(root.left)
        root.right = dfs(root.right)

        if not root.left and not root.right and root.val == target:
            root = None
        return root

    return dfs(root) 


class NumArray_BIT:
    # T: O(N) (cons: O(n); update/query: O(log(n)))
    # S: O(N)

    def __init__(self, nums: list[int]):
        # Fenwick Tree (Binary Indexed Tree, BIT)
        # 0 as dummy node, so BIT starts from 1
        self.size = len(nums)
        self.nums = nums[:]
        self.BIT = [0] + self.nums
        # this is slow, O(nlog(n))
        # self.BIT = [0] * (self.size + 1)
        # for i in range(self.size):
        #     self._construct(i+1, self.nums[i])
        for i in range(len(self.BIT)):
            next_node_idx = i + (i & -i)
            if next_node_idx < len(self.BIT):
                self.BIT[next_node_idx] += self.BIT[i]

    def _construct(self, index: int, val: int) -> None:
        # because we are at 0, no need to calc diff
        # get next: +
        idx = index
        while idx < len(self.BIT):
            self.BIT[idx] += val
            idx += idx & -idx
    
    def update(self, index: int, val: int) -> None:
        # focus on 增量
        # get next: +
        diff = val - self.nums[index]  # 計算增量
        self.nums[index] = val  # 更新原始數組
        idx = index + 1  # 轉換為 1-based 索引
        # while idx <= self.size:
        while idx < len(self.BIT):
            self.BIT[idx] += diff
            idx += idx & -idx
    
    def query(self, idx):
        # get parent, prefix sum: -
        # idx should be 1-indexed
        acc = 0
        while idx > 0:
            acc += self.BIT[idx]
            idx -= idx & -idx
        return acc
        
    def sumRange(self, left: int, right: int) -> int:
        # prefix sum
        # 1-indexed, so we add 1 for left & right
        return self.query(right+1) - self.query(left - 1 + 1)