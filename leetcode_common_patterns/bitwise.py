
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
