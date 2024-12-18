import math
from collections import OrderedDict
from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next  

def addBinary(a: str, b: str) -> str:

    a, b = a[::-1], b[::-1]
    ans = []
    carry = 0

    for i in range(max(len(a), len(b))):
        digit_a = int(a[i]) if i < len(a) else 0
        digit_b = int(b[i]) if i < len(b) else 0
        total = digit_a + digit_b + carry
        ans.append(str(total%2))
        carry = total // 2
    if carry:
        ans.append(str(carry))
    return ''.join(ans[::-1])

def addTwoNumbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:

    # total %
    # carry //

    cur_l1 = l1
    cur_l2 = l2
    carry = 0
    ans = ListNode()
    cur_ans = ans

    while cur_l1 and cur_l2:
        digit_l1 = cur_l1.val
        digit_l2 = cur_l2.val
        total = (digit_l1 + digit_l2 + carry)
        cur_ans.val = total % 10
        carry = total // 10
        cur_l1 = cur_l1.next
        cur_l2 = cur_l2.next

        if cur_l1 or cur_l2:
            cur_ans.next = ListNode()
            cur_ans = cur_ans.next

    while cur_l1:
        digit_l1 = cur_l1.val
        total = digit_l1 + carry
        cur_ans.val = total % 10 
        carry = total // 10
        cur_l1 = cur_l1.next
        if cur_l1:
            cur_ans.next = ListNode()
            cur_ans = cur_ans.next

    while cur_l2:
        digit_l2 = cur_l2.val
        total = digit_l2 + carry
        cur_ans.val = total % 10 
        carry = total // 10
        cur_l2 = cur_l2.next
        if cur_l2:
            cur_ans.next = ListNode()
            cur_ans = cur_ans.next
    if carry:
        cur_ans.next = ListNode(carry)
    return ans


def reverseBits(n: int) -> int:

    ret = 0

    for _ in range(32):
        lsb = n & 1
        ret = (ret << 1) | lsb
        n >>= 1
    return ret
        

def romanToInt(s: str) -> int:

    mp = {"I": 1, "V": 5, "X": 10,
            "L": 50, "C": 100, "D": 500, 
            "M": 1000 }
    
    ans = 0

    for i in range(len(s) - 1):
        if mp[s[i+1]] > mp[s[i]]:
            ans -= mp[s[i]]
        else:
            ans += mp[s[i]]
    return ans + mp[s[-1]]


class LRUCacheSimple:
    # LRU --> get and put will be treated as used
    # remove least one in case key not in LRU_CACHE and size > capacity
    # otherwise we set lru_cache[key] = value
    # production use this
    def __init__(self, capacity: int):
        self.lru_cache = OrderedDict()
        self.capacity = capacity
        

    def get(self, key: int) -> int:
        if key in self.lru_cache:
            self.lru_cache.move_to_end(key)
            return self.lru_cache[key]
        return -1
        

    def put(self, key: int, value: int) -> None:
        if key in self.lru_cache:
            self.lru_cache.move_to_end(key)
        else:
            if len(self.lru_cache) >= self.capacity:
                self.lru_cache.popitem(last=False)
        self.lru_cache[key] = value 


class Node:
    def __init__(self, key: int, val: int) -> None:
        # doubly linkedlist
        self.key = key
        self.val = val
        self.prev, self.next = None, None

class LRUCacheComplicatd:
    # interview use this
    # LRU --> get and put will be treated as used
    # remove least one in case key not in LRU_CACHE and size > capacity
    # otherwise we set lru_cache[key] = value
    # hashmap is for fast lookup O(1)
    # doublelinkedlist is for fast insert and deletion

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.lru_cache = dict() # key: Node
        self.left = Node(0, 0) # dummy
        self.right = Node(0, 0) # dummy
        self.left.next = self.right
        self.right.prev = self.left

    def remove(self, node):
        prev_node, next_node = node.prev, node.next
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def insert(self, node):
        prev_node, next_node = self.right.prev, self.right
        prev_node.next = node
        node.prev = prev_node
        node.next = next_node
        self.right.prev = node
        

    def get(self, key: int) -> int:
        if key in self.lru_cache:
            self.remove(self.lru_cache[key])
            self.insert(self.lru_cache[key])
            # emulate move_to_end
            return self.lru_cache[key].val
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.lru_cache:
            self.remove(self.lru_cache[key])
        self.lru_cache[key] = Node(key, value)
        self.insert(self.lru_cache[key])

        # remove LRU from the list
        # and key from hasmap
        if len(self.lru_cache) > self.capacity:
            lru = self.left.next
            self.remove(lru)
            del self.lru_cache[lru.key]
        

def reverse(x: int) -> int:
    INT_MAX = 2**31 - 1
    INT_MIN = -2**31
    MAX_LAST_DIGIT = int(math.fmod(INT_MAX, 10))
    MIN_LAST_DIGIT = int(math.fmod(INT_MIN, 10)) # cannot use % 10. Wrong
    INT_MAX_SAFE = int(INT_MAX / 10) 
    INT_MIN_SAFE = int(INT_MIN / 10) # cannot use //10. Wrong

    n = x
    ans = 0

    while n:
        last_digit = int(math.fmod(n, 10))
        n = int(n / 10)

        # >INT_MAX_SAFE + last_digit --> overflow
        # =INT_MAX_SAFE + > MAX_LAST_DIGIT --> overflow
        if ans > INT_MAX_SAFE or (ans == INT_MAX_SAFE and last_digit > MAX_LAST_DIGIT):
            return 0

        elif ans < INT_MIN_SAFE or (ans == INT_MIN_SAFE and last_digit < MIN_LAST_DIGIT):
            return 0
        ans = ans * 10 + last_digit
    return ans


def evalRPN(tokens: list[str]) -> int:

    stack = []
    ans = 0
    ops = {"+", "-", "*", "/"}

    if len(tokens) == 1:
        return tokens[0]

    for token in tokens:
        if token in ops:
            i1, i2 = stack.pop(), stack.pop()
            ans = int(eval(i2 + token + i1))
            stack.append(str(ans))
        else:
            # store numbers only
            stack.append(token)
    return ans



"""
x: 从起点到环入口的距离
y: 从环入口到相遇点的距离
z: 从相遇点绕环一圈回到环入口的距离


在第一次相遇时：

慢指针走的距离：x + y
快指针走的距离：x + y + n(y + z)，其中 n 是快指针绕环的圈数
因为快指针速度是慢指针的两倍，所以：

2(x + y) = x + y + n(y + z)
x + y = n(y + z)
x = (n-1)y + nz
x = (n-1)(y + z) + z
"""
def findDuplicate(nums: list[int]) -> int:

    # the key is the nums is in the range [1..n]

    slow = nums[0] # slow pointer
    fast = nums[0] # fast pointer

    while True:
        slow = nums[slow] # one step
        fast = nums[nums[fast]] # two step
        if slow == fast:
            # fast now is at meeting point in the cycle
            break
    # S to E: x
    # E to M: y
    # M to E: z
    # 2 * slow = fast
    # 2(x + y) =  x + y + n(y + z) 
    # where n = go through cycle n times

    # x = (n-1)(y+z) + z
    # fast goes (n-1)*(y+z) + z --> go back entry point
    # slow goes x --> go to the entry point
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    return slow
