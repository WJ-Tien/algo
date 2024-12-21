from typing import Optional
from bisect import bisect_left

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        slow = fast = head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        cur_head = slow
        prev = None

        while cur_head:
            next_node = cur_head.next
            cur_head.next = prev
            prev = cur_head
            cur_head = next_node
        reversed_head = prev
        # head --> ... --> null <-- slow ... <-- reversed_head 
        ori_head = head
        rev_head = reversed_head
        while rev_head.next: # very tricky !
            ori_next_node = ori_head.next # 2
            rev_next_node = rev_head.next # 3
        
            ori_head.next = rev_head # 4
            rev_head.next = ori_next_node # 2
        
            ori_head = ori_next_node # 2
            rev_head = rev_next_node # 3

def create_linked_list():
    # 建立節點
    node1 = ListNode(1)
    node2 = ListNode(2)
    node3 = ListNode(3)
    node4 = ListNode(4)
    
    # 連接節點
    node1.next = node2
    node2.next = node3
    node3.next = node4
    node4.next = None  # 最後一個節點指向 None
    
    return node1  # 返回頭節點

def print_linked_list(head):
    current = head
    while current:
        print(current.val, end="")
        if current.next:
            print(" -> ", end="")
        current = current.next
    print(" -> None")

# head = create_linked_list()
# s = Solution()
# s.reorderList(head)


nums = [10, 9, 8, 1, 2, 3]
ans = [100, 1]

for num in nums:
    pos = bisect_left(ans, num) 
    print(pos, num)
    if pos == len(ans):
        ans.append(num)
    else:
        ans[pos] = num

print(ans)
