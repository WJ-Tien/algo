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


def myPow(x: float, n: int) -> float:

    if n == 0:
        return 1.0
    elif n < 0:
        x = 1 / x
        n = -n
    
    ans = 1

    while n:
        if n % 2 == 1:
            ans *= x
        x *= x
        n //= 2
    return ans

def findMaxLength(nums: list[int]) -> int:
    # O(n) TS

    cur_sum = 0
    max_len = 0
    hmp = {0: -1}
    # 0: minus 1, 1: plus 1
    # if cur_sum == 0, we have balanced 0 & 1
    # we substract cur_index from hmp[cur_sum]

    for idx, num in enumerate(nums):
        if num == 0:
            cur_sum -= 1
        else:
            cur_sum += 1
        
        if cur_sum in hmp:
            # j = hmp[cur_sum]
            # i = cur_idx
            # j + 1 ~ i
            # i - (j + 1) + 1 = i - j
            max_len = max(max_len, idx - hmp[cur_sum])
        else:
            hmp[cur_sum] = idx
    return max_len