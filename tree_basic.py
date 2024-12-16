"""
pre-order:  <M> <L> <R>
in-order:   <L> <M> <R>
post-order: <L> <R> <M>
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



if __name__ == "__main__":
	t = BST()
	t.insert_node(100)
	t.insert_node(50)
	t.insert_node(110)
	t.insert_node(120)
	t.in_order_traversal(t.root)
	
