
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