

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


def decodeString(s: str) -> str:
    ans = []
    stack = []
    cur_sum = 0

    for char in s:
        if char.isdigit():
            cur_sum = cur_sum * 10 + int(char)
        elif char == "[":
            stack.append((ans, cur_sum))
            ans = []
            cur_sum = 0
        elif char == "]":
            prev_str, num = stack.pop()
            prev_str.extend(ans * num)
            ans = prev_str
        else:
            ans.append(char)
    return ''.join(ans)


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


def asteroidCollision(asteroids: list[int]) -> list[int]:

    stack = [] 

    for asteroid in asteroids:
        should_add = True
        while stack and stack[-1] > 0 and asteroid < 0: # critical
            prev_asteroid = stack[-1]
            if abs(prev_asteroid) < abs(asteroid):
                stack.pop()
                # critical
                # not yet to be added
                # need to check in another round
                continue
            elif abs(prev_asteroid) == abs(asteroid):
                stack.pop()
                should_add = False
                break
            else:
                should_add = False
                break

        if should_add:
            stack.append(asteroid)
    return stack
