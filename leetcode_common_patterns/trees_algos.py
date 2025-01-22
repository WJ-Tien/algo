
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

def lowestCommonAncestor_BT_OR_BSR(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    # T: O(N), last_line: O((logN)^2))
    queue = deque()
    queue.append(root)
    hmp = dict()
    hmp[root] = [root]

    while queue:
        node = queue.popleft()

        if node.left:
            queue.append(node.left)
            if node.left not in hmp:
                hmp[node.left] = []
            hmp[node.left].extend(hmp[node])
            hmp[node.left].append(node.left)
        if node.right:
            queue.append(node.right)
            if node.right not in hmp:
                hmp[node.right] = []
            hmp[node.right].extend(hmp[node])
            hmp[node.right].append(node.right)

    for p_dest in hmp[p][::-1]:
        for q_dest in hmp[q][::-1]:
            if p_dest.val == q_dest.val:
                return p_dest


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

