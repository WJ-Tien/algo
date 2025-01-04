

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



def calculate(s: str) -> int:
    # 227. basic calculator II
    # deal with * / immediately
    # for +/-, save it to stack only
    # store prev_op when char in +-*/ or idx == len(s) - 1

    num_stack = []
    cur_num = 0
    prev_op = "+"

    for idx, char in enumerate(s):
        if char.isdigit():
            cur_num = cur_num * 10 + int(char)
        
        if (char != " " and not char.isdigit()) or idx == len(s) - 1:
            if prev_op == "+":
                num_stack.append(cur_num)
            elif prev_op == "-":                    
                num_stack.append(-cur_num)
            elif prev_op == "*":
                num_stack.append(num_stack.pop() * cur_num)
            elif prev_op == "/":
                num_stack.append(int(num_stack.pop() / cur_num))
            cur_num = 0
            prev_op = char
    
    return sum(num_stack)


def basic_calculate(s: str) -> int:
    # 224 basic calculator
    """
    遇到數字 → 存起來
    遇到 +/- → 把之前數字加到結果中
    遇到 ( → 把目前結果存到 stack
    遇到 ) → 把 stack 的結果拿出來繼續算
    """

    stack = [] # store the previously calculated results (before ( )
    current_num = 0
    current_result = 0
    sign = 1

    for char in s:
        if char.isdigit():
            current_num = current_num * 10 + int(char) 

        elif char in {"+", "-"}:
            current_result += sign * current_num
            current_num = 0
            sign = 1 if char == "+" else -1
        elif char == "(":
            stack.append(current_result)
            stack.append(sign)
            current_result = 0
            sign = 1
        elif char == ")":
            current_result += sign * current_num
            current_num = 0
            current_result *= stack.pop()
            current_result += stack.pop()
    
    current_result += sign * current_num
    return current_result


def longestValidParentheses(s: str) -> int:
    # stack store indices
    # pop --> pop to match left
    # if stack is none, which means we just pop for right
    # current right's index as new starting point
    # calculate length

    stack = [-1]
    max_len = 0

    for i in range(len(s)): 
        if s[i] == "(":
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)
            max_len = max(max_len, i - stack[-1])
    return max_len


def largestRectangleArea(heights: list[int]) -> int:
    stack = [] # idx, height
    max_area = 0

    for i, cur_height in enumerate(heights):
        start = i
        while stack and stack[-1][1] > cur_height:
            # h(6) > h(2) --> stack.pop() --> 6 got popped
            idx, latest_height = stack.pop()
            max_area = max(max_area, latest_height * (i - idx))
            start = idx
        stack.append((start, cur_height)) # extend backward
    
    while stack:
        idx, latest_height = stack.pop()
        # can extend till the end (inclusive)
        max_area = max(max_area, latest_height * (len(heights) - idx))
    return max_area
 

class FreqStack:
    # T: O(1)
    # S: O(N)

    def __init__(self):
        self.freq = dict() # val: freq
        # very critical (stack of stack)
        # use freq to group cur val
        # while keep the order (top elements are placed later)
        self.groups = dict() # freq: [val1, val2 ...]

        self.max_freq = 0

    def push(self, val: int) -> None:
        self.freq[val] = self.freq.get(val, 0) + 1
        cur_freq = self.freq[val]
        if cur_freq >= self.max_freq:
            self.max_freq = cur_freq
        if cur_freq not in self.groups:
            self.groups[cur_freq] = [val]
        else:
            self.groups[cur_freq].append(val)

    def pop(self) -> int:
        ret = self.groups[self.max_freq].pop()
        self.freq[ret] -= 1
        if not self.groups[self.max_freq]:
            self.max_freq -= 1
        return ret