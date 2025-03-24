from collections import defaultdict

def minimizeXor(num1: int, num2: int) -> int:
    # 2429. Minimize XOR

    num2_ones = bin(num2).count("1")
    ans = 0

    # 32 bits

    # high bits first
    for i in range(31, -1, -1):
        if (num1 & (1 << i)) and num2_ones > 0:
            ans |= (1 << i)
            num2_ones -= 1

    # low bits
    for i in range(32):
        if num2_ones > 0 and not (ans & (1 << i)):
            ans |= (1 << i)
            num2_ones -= 1
    
    return ans


def hammingWeight(self, n: int) -> int:
    # 191. Number of 1 Bits (number of set bits)

    x = n
    ans = 0
    while x:
        ans += 1
        x &= (x-1)
    
    return ans

def countBits(n: int) -> list[int]:
    # 338. Counting Bits
    # i & (i-1) --> i & (i - 1) 是 i 移除最低位 1 之後的數字。
    # include 1
    # i &= (i - 1) 會將 i 最低的 1 變成 0，並且將這個 1 之後的所有 0 保持不變

    ans = [0] * (n+1)

    for i in range(1, n+1):
        ans[i] = ans[i&(i-1)] + 1
    return ans


def duplicateNumbersXOR(nums: list[int]) -> int:
    # 3158. Find the XOR of Numbers Which Appear Twice
    # len(nums) <= 50 --> can fit 64 bits
    ans = 0
    seen = 0
    for num in nums:
        if seen & (1 << num):
            ans ^= num
        else:
            seen |= (1 << num)
    return ans


def isPowerOfTwo(n: int) -> bool:
    #   n = 100
    # n-1 = 011
    return n != 0 and n & (n-1) == 0
    # return (x & -x) == x
    # x & -x get the rightmost set bit, and to set all the other bits to 0.



def countTriplets(arr: list[int]) -> int:
    # i < j <= k
    # 2 = 3 ^ 1
    ans = 0
    prefix = 0
    count_map = defaultdict(int)
    count_map[0] = 1
    indices_sum = defaultdict(int)

    for i in range(len(arr)):
        prefix ^= arr[i]
        # 第一部分 XOR（arr[i] ^ ... ^ arr[j-1]）
        # 等於第二部分 XOR（arr[j] ^ ... ^ arr[k]）
        # prefix[j-1] ^ prefix[i-1] = prefix[k] ^ prefix[j-1]
        # prefix[i-1] = prefix[k]
        # 這意味著，只要我們找到兩個位置 i-1 和 k，使得它們的前綴 XOR 值相等
        #（prefix[i-1] = prefix[k]），那麼中間的 j（滿足 i < j <= k）
        # 就可以構成一個有效的三元組。
        # sum(k - i) = k + k + ... + k（n 次） - (i1 + i2 + ... + ik)
        ans += count_map[prefix] * i - indices_sum[prefix]
        indices_sum[prefix] += i + 1
        count_map[prefix] += 1
    return ans