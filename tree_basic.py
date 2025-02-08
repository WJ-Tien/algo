"""
pre-order:  <M> <L> <R>
in-order:   <L> <M> <R>
post-order: <L> <R> <M>
"""

"""
DFS: 543. Diameter of Binary Tree
BFS: 102. Binary Tree Level Order Traversal 
N-ary: 589. N-ary Tree Preorder Traversal (DFS)

"""

# AVL > Red-black >> B/BST

class Node:
	def __init__(self, value, left=None, right=None):
		self.value = value
		self.left = left
		self.right = right

class BST:

	""" Binary Search Tree """ 

	def __init__(self):
		self._root = None

	@property
	def root(self):
		return self._root

	def _insert_node_helper(self, node, value):
			
		if node is None:
			if self._root is None:
				self._root = Node(value)
			else:
				node = Node(value)
		else:
			cur = self._root
			if cur.value > value:
				if node.left is None:
					node.left = Node(value)
				else:
					self._insert_node_helper(node.left, value)
			else:	
				if node.right is None:
					node.right = Node(value)
				else:
					self._insert_node_helper(node.right, value)

	def insert_node(self, value):
		self._insert_node_helper(self._root, value)

	def delete_node(self, value):
		# TODO
		return 

	def in_order_traversal(self, node):
		""" dfs """
		if node is None:
			return None

		print(node.value)
		self.in_order_traversal(node.left)
		self.in_order_traversal(node.right)

class FenwickTree:
	# https://www.youtube.com/watch?v=CWDQJGaN1gY&ab_channel=TusharRoy-CodingMadeSimple
	# To construct the whole tree (prefix_sum) -> O(n*log(n))
    def __init__(self, size):
		# S: O(N)
        """ 初始化 Fenwick Tree，建立一個長度為 size+1 的陣列（索引從 1 開始） """
        self.size = size
        self.tree = [0] * (size + 1) # 0 = dummy node arr[i+1] = bit[i]
		# update idx i+1 in BIT using idx i in the arr
		# for i in range(self.size):
        #     self.update(i + 1, arr[i])  # 注意：Fenwick Tree 索引從 1 開始

		# batch -> O(n) instead of O(nlogn)
		# self.tree = [0] + arr[:]  # 索引從 1 開始
        # for i in range(1, self.size + 1):
        #     j = i + (i & -i)  # 找到下一個需要影響的索引
        #     if j <= self.size:
        #         self.tree[j] += self.tree[i]  # 直接累加	

    def update(self, idx, val):
		# T: O(logN)
		# get next
        """ 在索引 idx 加上 val，並更新相關節點 """
		# idx += 1
        while idx <= self.size:
            self.tree[idx] += val
            idx += idx & -idx  # move to the next node

    def query(self, idx):
		# T: O(logN)
		# get parent
        """ 查詢從 1 到 idx 的前綴和 """
		# idx += 1
        sum_ = 0
        while idx > 0:
            sum_ += self.tree[idx]
            idx -= idx & -idx  # 移動到前一個影響範圍內的節點
        return sum_

    def range_query(self, left, right):
		# T: O(logN)
        """ 查詢範圍 [left, right] 內的總和 """
        return self.query(right) - self.query(left - 1)

# # 測試範例
# fenwick = FenwickTree(10)  # 建立大小為 10 的 Fenwick Tree
# fenwick.update(3, 5)  # 在索引 3 加上 5
# fenwick.update(5, 7)  # 在索引 5 加上 7
# fenwick.update(7, 10) # 在索引 7 加上 10

# print(fenwick.query(5))  # 查詢前 5 個元素的總和，輸出 12 (5 + 7)
# print(fenwick.range_query(3, 7))  # 查詢索引 3 到 7 的總和，輸出 22
# for non-complete graph we can set i-j (not exist) to +inf

class SegmentTree:
    """
        在 Segment Tree 中，我們根據 數組的索引範圍 劃分：
        左子樹存儲 左半區間 的資訊。
        右子樹存儲 右半區間 的資訊。 
        不斷切半 直到剩下一個
    
    """

    def __init__(self, nums: list[int]):
        # T: average O(logN); build_tree O(N)
        # S: O(4*N) = O(N)
        # h = log_2(N)
        # number of nodes = 2^(h+1) - 1
        # 4N can guarantee enough capacity
        self.n = len(nums)
        self.tree = [0] * (4 * self.n) # 4N
        self.build_tree(nums, 0, 0, self.n-1)
        
    def build_tree(self, arr, node, start , end):
        # T: O(n)
        # node (idx) -> from tree
        # start/end (idx) -> from arr
        # build prefix_sum tree
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = start + (end - start) // 2
            left_node = 2 * node + 1
            right_node = 2 * node + 2
            self.build_tree(arr, left_node, start, mid)
            self.build_tree(arr, right_node, mid+1, end)
            self.tree[node] = self.tree[left_node] + self.tree[right_node]

    def _update(self, node, start, end, index, val):
        # T: O(logN), worst O(N)
        if start == end:
            self.tree[node] = val
        else:
            mid = start + (end - start) // 2
            left_node = 2 * node + 1
            right_node = 2 * node + 2
            if index <= mid:
                self._update(left_node, start, mid, index, val)
            else:
                self._update(right_node, mid+1, end, index, val)
            self.tree[node] = self.tree[left_node] + self.tree[right_node] 

    def update(self, index: int, val: int) -> None:
        self._update(0, 0, self.n-1, index, val)
    
    def _query(self, node, start, end, L, R):
        # T: O(logN)
        # L...R start...end L...R
        if R < start or L > end:
            return 0
        elif L <= start and end <= R:
            return self.tree[node]
        else:
            mid = start + (end - start) // 2
            left_node = 2 * node + 1
            right_node = 2 * node + 2
            left_sum = self._query(left_node, start, mid, L, R)
            right_sum = self._query(right_node, mid + 1, end, L, R)
            return left_sum + right_sum

    def sumRange(self, left: int, right: int) -> int:
        # T: O(logN)
        return self._query(0, 0, self.n - 1, left, right)




if __name__ == "__main__":
	t = BST()
	t.insert_node(100)
	t.insert_node(50)
	t.insert_node(110)
	t.insert_node(120)
	t.in_order_traversal(t.root)
	
