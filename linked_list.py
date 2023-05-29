class Node:
	def __init__(self, value, next_ptr=None):
		self.value = value
		self.next_ptr = next_ptr

class LinkedList:
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

	def reversed_list(self):
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
	L = LinkedList()
	L.push_back_node(1)
	L.insert_front_node(2)
	L.push_back_node(3)
	print(L.get_size())
	L.print_node()
	L.reversed_list()
	L.print_node()
