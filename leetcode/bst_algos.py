from collections import deque
from typing import Optional

# tree template

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def isBalanced_recur(root: Optional[TreeNode]) -> bool:
    
    def dfs(root):
        if root is None:
            return (True, 0) # (isBalanced, cur_max_height)
        
        lh = dfs(root.left)
        rh = dfs(root.right)

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
                # cal height
                left_depth = depth.get(node.left, 0)
                right_depth = depth.get(node.right, 0)

                if abs(left_depth - right_depth) > 1:
                    return False

                depth[node] = 1 + max(left_depth, right_depth)

        return True