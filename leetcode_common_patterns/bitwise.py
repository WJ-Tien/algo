
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
    # 1442. Count Triplets That Can Form Two Arrays of Equal XOR

    # O(N) TS

    # Suppose index "a, b, c" all has the same prefix xor. 
    # Current index is "i", then the result is:
    # (i-a-1)+(i-b-1)+(i-c-1) == i*3-(a+b+c)-3.
    # i*cnt - (sum(idx)) - cnt
    # X(i, k) = 0 
    # i+1 (i), i+2~k (j), k (k) 
    # hence the #comb is  k- i - 1

    # @ idx = -1, prefix_xor_sum = 0, and zero has one count
    prefix_xor_sum = {0: [-1, 1]} # prefix_xor: [idxSum, cnt]
    ans = 0
    cur_xor_sum = 0

    for idx, num in enumerate(arr):
        cur_xor_sum ^= num
        idx_sum, cnt = prefix_xor_sum.get(cur_xor_sum, [0, 0])
        ans += ((idx * cnt) - idx_sum - cnt)
        prefix_xor_sum[cur_xor_sum] = [idx_sum + idx, cnt + 1]
    return ans