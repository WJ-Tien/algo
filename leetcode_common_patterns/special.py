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


def leastInterval(tasks: list[str], n: int) -> int:
    # 621 task scheduler

    fhmp = dict()
    for task in tasks:
        fhmp[task] = fhmp.get(task, 0) + 1
    
    max_freq = max(fhmp.values()) 
    max_count = 0
    for value in fhmp.values():
        if max_freq == value:
            max_count += 1
    
    # max_freq - 1 interval * (n + 1) + max_count (remaining)
    # e.g. n = 2
    # AB_|AB_|AB
    ideal_block_len = (max_freq - 1) * (n + 1) + max_count

    return max(ideal_block_len, len(tasks))


def myAtoi(s: str) -> int:
    if not s:
        return 0
    
    index = 0
    n = len(s)
    while index < n and s[index] == ' ':
        index += 1
    
    # 如果都是空白，略過後就沒字元了，直接回傳 0
    if index == n:
        return 0
    
    # 3. 準備正負號、結果變數，以及上下限
    sign = 1
    result = 0
    INT_MAX = 2**31 - 1
    INT_MIN = -2**31
    
    # 4. 檢查第一個非空白字元是否為正負號
    if s[index] == '-':
        sign = -1
        index += 1
    elif s[index] == '+':
        index += 1
    
    # 5. 開始讀取後續數字
    while index < n and s[index].isdigit():
        digit = int(s[index])
        
        if (result > INT_MAX // 10) or \
            (result == INT_MAX // 10 and digit >= 8):
            return INT_MAX if sign == 1 else INT_MIN
        
        # 7. 組合數字
        result = result * 10 + digit # implicity exclude the leading zero
        index += 1
    
    # 8. 根據正負號回傳最終數值
    return sign * result
    


class LargerNumStr(str):
    # 179. Largest Number
    # 當 Python 排序需要判斷 self < other 時，就會呼叫這個方法
    def __lt__(self, other):
        # 若 self+other > other+self，表示在題目的意義下 "self" 應該放前面
        # 因為我們想把「拼起來大的」排在前面，
        # 但 Python 的 sort() 是把 "比較小" 的放前面，
        # 所以這裡就直接反過來寫：x < y if x+y > y+x
        # x + y > y + x --> indicates x < y (true for __lt__)
        return self + other > other + self

class Solution:
    def largestNumber(self, nums: list[int]) -> str:
        # 179. Largest Number

        nums = [LargerNumStr(str(num)) for num in nums]
    
        # 排序 (預設 ascending)，但因為我們把 __lt__ 反轉過來，實際會把「應放前面的」排在前面
        nums.sort()
        
        # 若排完之後第一個字是 '0'，代表全部都是零
        if nums[0] == '0':
            return '0'
        
        return ''.join(nums)
        
    
class Codec:
    def encode(self, strs: list[str]) -> str:
        """Encodes a list of strings to a single string.
        """
        enc = []
        for s in strs:
            enc.append(str(len(s)))
            enc.append("/:")
            enc.append(s)
        return ''.join(enc)


    def decode(self, s: str) -> list[str]:
        """Decodes a single string to a list of strings.
        """
        dec = []
        i = 0

        # "5/:Hello5/:World"
        # T: amortized O(n)
        while i < len(s):
            delim = s.find("/:", i)
            length = int(s[i:delim])
            dec.append(s[delim+2:delim+2+length])
            i = delim + 2 + length
        return dec