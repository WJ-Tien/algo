
def longestPalindrome(s: str) -> int:
    hmp = dict()
    odd = 0
    acc = 0
    
    for char in s:
        if char not in hmp:
            hmp[char] = 1
        else:
            hmp[char] += 1
    for val in hmp.values(): 
        if val % 2:
            acc += (val - 1) 
            odd += 1
        else:
            acc += val
    if odd > 0:
        acc += 1
    return acc

def longestCommonPrefix(strs: list[str]) -> str:

    ans = []

    for i in range(len(strs[0])):
        for s in strs[1:]:
            if i == len(s) or s[i] != strs[0][i]:
                return ''.join(ans)
        ans += strs[0][i]
    return ''.join(ans)

def lengthOfLongestSubstring(self, s: str) -> int:
    # 3. Longest Substring Without Repeating Characters
    # sliding window + hashmap
    """
    如果當前字符已經存在於 str_map，且它的索引位置大於或等於 start，表示當前窗口內發生重複。
    將 start 更新為 str_map[chars] + 1，也就是從重複字符的下一個位置開始新的窗口。
    更新字符位置：

    無論是否有重複字符，將當前字符的索引存入 str_map，表示它的最新位置。
    """
    start = 0
    max_len = float("-inf") 
    hmp = dict()

    for idx, char in enumerate(s):
        if char in hmp and hmp[char] >= start:
            start = hmp[char] + 1
        hmp[char] = idx
        max_len = max(max_len, idx - start + 1)
    return max_len

