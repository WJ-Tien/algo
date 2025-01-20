from collections import deque


def nextGreaterElement(nums1: list[int], nums2: list[int]) -> list[int]:

    hmp = dict()
    for idx, num in enumerate(nums2):
        hmp[num] = idx
    
    stack = [] # idx
    ans = [-1] * len(nums2)

    for idx, num in enumerate(nums2):
        while stack and nums2[stack[-1]] < num:
            prev_idx = stack.pop()
            ans[prev_idx] = num 
        stack.append(idx)

    return [ans[hmp[num]] for num in nums1]


def nextGreaterElementsII(nums: list[int]) -> list[int]:

    # simulate circular
    extended_nums = nums * 2
    stack = []
    ans = [-1] * len(nums)
    for idx, num in enumerate(extended_nums):
        while stack and extended_nums[stack[-1]] < num:
            prev_idx = stack.pop()
            ans[prev_idx] = num
        # we only consider idx < len(nums)
        # since it's symmetric
        if idx < len(nums):
            stack.append(idx)
     
        return ans

def dailyTemperatures(temperatures: list[int]) -> list[int]:

    # stack stores idx
    # if stack[-1] < temperatures
    # start to pop and calculate day diff
    # monotonically desc

    stack = []
    ans = [0] * len(temperatures) 

    for idx, temperature in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < temperature:
            prev_idx = stack.pop()
            ans[prev_idx] = idx - prev_idx
        
        stack.append(idx)
    return ans


def finalPrices(prices: list[int]) -> list[int]:
    # 1475. Final Prices With a Special Discount in a Shop

    ans = prices.copy()
    stack = []

    for idx, price in enumerate(prices):
        while stack and prices[stack[-1]] >= price:
            prev_idx = stack.pop()
            ans[prev_idx] = prices[prev_idx] - price
        stack.append(idx)
    
    return ans 


class StockSpanner:
    # 901. Online Stock Span

    def __init__(self):
        self.stack = [] # (price, span)

    def next(self, price: int) -> int:
        cur_span = 1 # default 1
        while self.stack and self.stack[-1][0] <= price:
            prev_price, prev_span = self.stack.pop()
            cur_span += prev_span
        
        self.stack.append((price, cur_span))
        
        return self.stack[-1][-1]


def maxSlidingWindow(nums: list[int], k: int) -> list[int]:

    # max(window) = overall O(n^2) --> TLE
    # we use monotonic queue to make the whole algo O(n)
    # pop/popleft first and finally get max

    ans = []
    queue = deque() # monotonic desc queue; store idx

    for idx, num in enumerate(nums):

        # trace max value
        while queue and nums[queue[-1]] < num: 
            queue.pop()

        queue.append(idx)
        
        # out-of-bound
        if queue[0] <= idx - k:
            queue.popleft()
        
        # every k, we can calculate a max
        # since each iteration we increase idx by 1
        # so we guarantee at each iteration
        # we can calculate a max
        # so we use >= k - 1 (0-indexed, we get a k size window)
        if idx >= k - 1:
            ans.append(nums[queue[0]])
        
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


def canBeValid_opt(s: str, locked: str) -> bool:
    # stack S: O(n)
    # this sol: S: O(1)
    """
    檢查在含有未鎖定字符 (locked[i] == '0') 的情況下，
    字串 s 能否透過適當地將未鎖定字符視為 '(' 或 ')'，
    成為有效的括號序列。
    """

    # 1. 長度為奇數，不可能成為有效括號，直接 False
    if len(s) % 2 != 0:
        return False

    # ---------------------------
    #  2. 前向遍歷：檢查「右括號是否過多」
    #  目的：確保從左到右的任何位置都不會出現 ')' 超過 '(' 的狀況
    #  作法：將「未鎖定字符」優先視為 '(' (balance += 1)
    # ---------------------------
    balance = 0
    for i in range(len(s)):
        if locked[i] == '0' or s[i] == '(':
            # locked[i] == '0' 表示未鎖定，當成 '('；或本身就是 '('
            balance += 1
        else:
            # 剩下只能是 locked[i] == '1' 且 s[i] == ')'
            balance -= 1

        # 隨時檢查：若 balance < 0 表示 ')' 過多
        if balance < 0:
            return False

    # ---------------------------
    #  3. 後向遍歷：檢查「左括號是否過多」
    #  目的：確保從右到左的任何位置都不會出現 '(' 超過 ')' 的狀況
    #  作法：將「未鎖定字符」優先視為 ')' (balance += 1)
    # ---------------------------
    balance = 0
    for i in range(len(s) - 1, -1, -1):
        if locked[i] == '0' or s[i] == ')':
            # locked[i] == '0' 表示未鎖定，當成 ')'；或本身就是 ')'
            balance += 1
        else:
            # 剩下只能是 locked[i] == '1' 且 s[i] == '('
            balance -= 1

        # 若 balance < 0 表示 '(' 過多（在後向視角下）
        if balance < 0:
            return False

    # 4. 若前、後向遍歷都沒問題，代表字串可以成為有效括號
    return True


def canBeValid_stack(s: str, locked: str):
    # 921. Minimum Add to Make Parentheses Valid
    # greedy algo
    length = len(s)

    # If length of string is odd, return false.
    if length % 2 == 1:
        return False

    open_brackets = []
    unlocked = []

    # Iterate through the string to handle '(' and ')'
    for i in range(length):
        if locked[i] == "0":
            unlocked.append(i)
        elif s[i] == "(":
            open_brackets.append(i)
        elif s[i] == ")":
            if open_brackets:
                open_brackets.pop()
            elif unlocked:
                unlocked.pop()
            else:
                return False

    # Match remaining open brackets and the unlocked characters
    while open_brackets and unlocked and open_brackets[-1] < unlocked[-1]:
        open_brackets.pop()
        unlocked.pop()

    if open_brackets:
        return False

    return True


def minAddToMakeValid(s: str) -> int:
    # 921. Minimum Add to Make Parentheses Valid
    stack = []
    ans = 0

    for char in s:
        if char == "(":
            stack.append(")")
        else:
            if stack:
                stack.pop()
            else:
                ans += 1
    if stack:
        ans += len(stack)
    return ans



def simplifyPath(path: str) -> str:

    stack = []
    path_sp = path.split("/")

    for path in path_sp:
        # .. start
        if not stack and path == "..":
            continue
        elif stack and path == "..":
            stack.pop()
        else:
            if path not in {".", ""}:
                stack.append(path)

    return "/" + '/'.join(stack)
