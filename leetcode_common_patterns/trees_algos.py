from collections import deque
from typing import Optional

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
    
    