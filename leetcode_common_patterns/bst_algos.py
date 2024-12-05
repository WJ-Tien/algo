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