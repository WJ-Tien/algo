class Node:
	def __init__(self, value, next_ptr=None):
		self.value = value
		self.next_ptr = next_ptr

class SinglyLinkedList:
	def __init__(self):
		self.head = None
		self.size = 0

	def insert_front_node(self, value):
		if self.head is None:
			self.head = Node(value)
		else:
			cur_head = self.head
			self.head = Node(value)
			self.head.next_ptr = cur_head

		self.size += 1

	def push_back_node(self, value):
		if self.head is None:
			self.head = Node(value)
		else:
			cur_head = self.head
			prev = None
			while cur_head:
				prev = cur_head	
				cur_head = cur_head.next_ptr
			prev.next_ptr = Node(value)

		self.size += 1

	def print_node(self):
		cur_head = self.head

		while cur_head:
			print(cur_head.value, end=" -> ")
			cur_head = cur_head.next_ptr
		print("null")

	def get_size(self):
		return self.size
	
	def delete_node_at(self, node_index):
		list_size = self.get_size()

		if list_size - 1 < node_index or node_index < 0:
			raise IndexError("invalid index")

		cur_head = self.head
		cur_index = 0
		prev = None
		while cur_head:
			if cur_index == node_index:
				self.size -= 1
				if node_index == 0:
					self.head = self.head.next_ptr
					return
				to_delete_next = cur_head.next_ptr	
				prev.next_ptr = to_delete_next
				return 
			prev = cur_head
			cur_head = cur_head.next_ptr
			cur_index += 1

	def reverse_list(self):
		cur_head = self.head
		prev = None
		
		while cur_head:
			# record cur next
			# point to prev 
			# mod prev as cur
			# move cur to the cur next
			the_next = cur_head.next_ptr
			cur_head.next_ptr = prev
			prev = cur_head
			cur_head = the_next

		self.head = prev	

if __name__ == "__main__":
	SLL = SinglyLinkedList()
	
	SLL.push_back_node(1)
	SLL.print_node()
	SLL.insert_front_node(2)
	SLL.push_back_node(3)
	SLL.delete_node_at(2)
	SLL.print_node()
	SLL.reverse_list()
	SLL.print_node()
	print(SLL.get_size())
