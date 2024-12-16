

def dailyTemperatures(temperatures: list[int]) -> list[int]:

    # O(n) TS

    ans = [0] * len(temperatures)
    stack = [] # store (temp, idx)

    for idx, temperature in enumerate(temperatures):
        # key using while since we might have multiples 
        # that is smaller than this temperature
        while stack and temperature > stack[-1][0]:
            stack_temp, stack_idx = stack.pop()
            ans[stack_idx] = idx - stack_idx
        stack.append((temperature, idx))
    return ans

        


