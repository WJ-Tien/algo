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
        

def productExceptSelf(nums: list[int]) -> list[int]:
    # prefix product like
    ret = [1] * len(nums)

    # ret[i-1] = prefix_prod before i - 1 (0~i-2)
    # prod[0~i) = prefix_prod[0 ~ i - 2] * nums[i-1]

    for i in range(1, len(nums)):
        ret[i] = ret[i-1] * nums[i - 1]
    
    rprod = 1
    for i in range(len(nums)-1, -1, -1):
        ret[i] *= rprod # the order of this two lines is critical
        rprod *= nums[i]

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