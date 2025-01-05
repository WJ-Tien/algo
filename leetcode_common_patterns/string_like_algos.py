from bisect import bisect_left
from collections import Counter

"""
Tips: hashmap + sliding window, and two_pointers
Also Put all the longest OOO here
"""
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

def lengthOfLongestSubstring(s: str) -> int:
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


def lengthOfLIS(nums: list[int]) -> int:
    # 300. Longest Increasing Subsequence
    ans = []
    
    # bisect --> return len(arr) if pos not in arr
    # otherwise return existing pos
    for num in nums:
        pos = bisect_left(ans, num) # arr must be sorted
        if pos == len(ans):
            ans.append(num)
        else:
            ans[pos] = num
    
    return len(ans)


def longestConsecutive(nums: list[int]) -> int:
    # longest consecutive sequence

    max_len = 0
    cur_len = 0
    num_set = set(nums)

    for num in num_set:
        if num - 1 not in num_set:
            # start point
            cur_num = num
            cur_len = 1
            while cur_num + 1 in num_set:
                cur_num += 1
                cur_len += 1 
        
            max_len = max(max_len, cur_len)
    return max_len


def longestPalindrome(s: str) -> str: #noqa

    # expand algo
    # T: O(n^2)
    # S: O(n)
    max_len = 0
    start = 0
    n = len(s)

    # odd part
    # a b c b a
    for mid in range(n):
        x = 0
        while mid - x >= 0 and mid + x < n:
            if s[mid - x] != s[mid + x]:
                break
            cur_len = 2 * x + 1 # 1 for center
            if cur_len > max_len:
                max_len = cur_len
                start = mid - x 
            x += 1
    
    # even part
    for mid in range(n):
        x = 1
        # ab . ba
        # mid . mid + 1
        # mid - 1 .. mid + 2
        while mid - x + 1 >= 0 and mid + x < n:
            if s[mid - x + 1] != s[mid + x]:
                break
            cur_len = 2 * x
            if cur_len > max_len:
                max_len = cur_len
                start = mid - x + 1
            x += 1
    
    return s[start: start + max_len]

def characterReplacement(s: str, k: int) -> int:
    # Longest Repeating Character Replacement

    start = 0
    hmp = dict()
    max_freq = 0
    max_len = 0

    for end in range(len(s)):
        hmp[s[end]] = hmp.get(s[end], 0) + 1

        max_freq = max(max_freq, hmp[s[end]])

        can_be_replaced_by_k = end - start + 1 - max_freq <= k
        if not can_be_replaced_by_k:
            # move start by 1
            hmp[s[start]] -= 1
            start += 1
        
        max_len = max(max_len, end - start + 1)
    return max_len


def findAnagrams(s: str, p: str) -> list[int]:
    # 438. Find All Anagrams in a String
    # sliding windows + hashmap
    ns, np = len(s), len(p)
    if ns < np:
        return []

    p_count = Counter(p)
    s_count = Counter()

    ans = []

    for i in range(ns):
        s_count[s[i]] += 1

        if i >= np:
            if s_count[s[i-np]] == 1:
                del s_count[s[i-np]]
            else:
                s_count[s[i-np]] -= 1

        if p_count == s_count:
            ans.append(i + 1 - np)
    return ans


def palindromePairs(words: list[str]) -> list[list[int]]:
    # T: O(n*k^2)
    # S: O(n*k + n^2)
    # 建立字典，存儲每個字串的反轉及其索引
    word_dict = {word[::-1]: i for i, word in enumerate(words)}
    result = []

    for i, word in enumerate(words):
        for j in range(len(word) + 1):  # 分割點，包含整個字串和空字串
            left, right = word[:j], word[j:]
            
            # 如果左邊是回文，檢查右邊的反轉是否在字典中且索引不同
            # left = ""
            # right = "abc"
            # 要讓 "" + "abc" 變回文，得在「整個字串」左邊補上 "abc" 的反轉 → "cba"。
            # "cba" + "" + "abc"
            if left == left[::-1] and right in word_dict and word_dict[right] != i:
                result.append([word_dict[right], i])
            
            # 如果右邊是回文，檢查左邊的反轉是否在字典中且索引不同
            # 確保避免重複，當右邊長度不為 0 時才檢查
            # j = len(word)，left = word（整個字串）
            if j != len(word) and right == right[::-1] and left in word_dict and word_dict[left] != i:
                result.append([i, word_dict[left]])

    return result