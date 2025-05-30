from bisect import bisect_left
from collections import Counter, deque

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

    假設 s = "abba"，當我們處理到最後一個 a 時：
    此時，start = 2，right = 3，而 hmp['a'] = 0（第一個 a 的位置）。
    如果沒有檢查 hmp[s[right]] >= start，我們會錯誤地將 start 移動到 hmp['a'] + 1 = 1，但其實 start 已經在 1 後面了，這樣會影響子字串的正確性。
    只有當 hmp[s[right]] 大於等於 start 時，才需要更新 start。
    """
    start = 0
    max_len = float("-inf") 
    hmp = dict()

    for idx, char in enumerate(s):
        # we need "hmp[char] >= start"
        # because start might go faster than hmp[char]
        # e.g., abba case
        # only within valid range, we update start
        # for =, we can see abca
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


def stringMatching(words: list[str]) -> list[str]:
    # 1408. String Matching in an Array
    # O(m*n^2), m = average word length
    # O(m*n)
    # Longest Proper Prefix which is also Suffix
    """
    前綴(Prefix)和後綴(Suffix)的定義：
    前綴：字串從開頭到某個位置的子字串（不包含完整字串本身）
    後綴：字串從某個位置到結尾的子字串（不包含完整字串本身）
    以字串 "ABAB" 為例：
    前綴集合：{"A", "AB", "ABA"}
    後綴集合：{"B", "AB", "BAB"}
    共同前後綴就是同時出現在前綴集合和後綴集合中的字串。
    在這個例子中，"AB" 就是一個共同前後綴。
    """


    def kmp_lps(pattern):
        """
        建立 LPS 表，用於加速 KMP 演算法。
        """
        lps = [0] * len(pattern)
        length = 0  # 目前的部分匹配長度
        i = 1
    
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
    
        return lps
    
    def kmp_search(text, pattern):
        """
        使用 KMP 演算法檢查 pattern 是否為 text 的子字串。
        """
        lps = kmp_lps(pattern)
        i = 0  # text 的索引
        j = 0  # pattern 的索引
    
        while i < len(text):
            if text[i] == pattern[j]:
                i += 1
                j += 1
    
            if j == len(pattern):  # 完整匹配
                return True
            elif i < len(text) and text[i] != pattern[j]:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
    
        return False

    ans = []
    for i in range(len(words)):
        for j in range(len(words)):
            if i != j and kmp_search(words[j], words[i]):
                ans.append(words[i])
                break

    return ans


def wordSubsets(words1: list[str], words2: list[str]) -> list[str]:
    # T: O(N2*k + N1*L)
    # S: O(26)
    # we only need to collect max freq
    # as long as you meet max_freq
    # then the remaining chars will fit 
    max_freq = Counter()
    for word in words2:
        cur_word = Counter(word)
        for char, count in cur_word.items():
            max_freq[char] = max(max_freq[char], count)
    
    ans = []
    for word in words1:
        cur_word = Counter(word)
        flag = True
        for char, count in max_freq.items():
            if cur_word[char] < max_freq[char]:
                flag = False
                break
        if flag:
            ans.append(word)
    return ans


def maxScore(s: str) -> int:
    # 1422. Maximum Score After Splitting a String
    """
    (左邊0的數量 + 右邊1的數量) 
        = (左邊0的數量 + 總1的數量 - 左邊1的數量)
        = (左邊0的數量 - 左邊1的數量) + 總1的數量
    """
    ones = 0
    zeroes = 0
    ans = float("-inf")

    for num in s[:-1]:
        if num == "1":
            ones += 1
        else:
            zeroes += 1
        
        ans = max(ans, zeroes - ones)
    
    if s[-1] == "1":
        ones += 1
    return ans + ones


def backspaceCompare(s: str, t: str) -> bool:
    # T: O(N)
    # O(N) space using stack is trivial
    # O(1) space solution is tricky 

    def next_valid_idx(string, idx):
        skip = 0

        # moving backward allow us to use only
        # skip vairiable to trace the #
        # without using two stacks
        # tricky but amazing solution here

        while idx >= 0:
            if string[idx] == "#":
                skip += 1
            elif skip > 0: 
                # valid char
                skip -= 1
            else: 
                # valid char and skip == 0
                return idx
            idx -= 1
        return -1
    
    i = len(s) - 1
    j = len(t) - 1

    # using and here cause errors a, a#a
    while i >= 0 or j >= 0: 
        i = next_valid_idx(s, i)
        j = next_valid_idx(t, j)

        if i >= 0 and j >= 0 and s[i] != t[j]:
            return False

        # check if both of them has remaining chars 
        # within valid range
        if (i >= 0) != (j >= 0):
            return False
        i -= 1
        j -= 1

    return True


def isPalindrome_i(s: str) -> bool:
    # 125. Valid Palindrome

    l, r = 0, len(s) - 1 #noqa

    # negative lists

    while l <= r:
        if not s[l].isalnum():
            l += 1 # noqa
            continue

        if not s[r].isalnum():
            r -= 1
            continue
        
        if s[l].lower() != s[r].lower():
            return False
        
        r -= 1
        l += 1 #noqa
    return True


def validPalindromeII(s: str) -> bool:
    # 680. Valid Palindrome II

    def is_palindrome(left: int, right: int) -> bool:
        while left <= right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        
        return True
    
    left, right = 0, len(s) - 1 
    while left <= right:
        if s[left] != s[right]:
        # either remove left or remove right
            return is_palindrome(left+1, right) or is_palindrome(left, right-1)
        left += 1
        right -= 1
    return True


def gcdOfStrings(str1: str, str2: str) -> str:
    # 1071. Greatest Common Divisor of Strings
    # O(n+m) TS

    def gcd(a, b):
        # O(log(min(m,n)))
        while b:
            a, b = b, a % b
        return a
    
    # O(n + m)
    if str1 + str2 != str2 + str1:
        return ""
    
    max_len = gcd(len(str1), len(str2))
    return str1[:max_len]


def isIsomorphic(s: str, t: str) -> bool:
    # 205. Isomorphic Strings
    
    if len(s) != len(t):
        return False

    def iso_group(s: str) -> list[int]:
        hmp = dict()
        idx = 0
        iso = []

        for char in s:
            if char not in hmp:
                hmp[char] = idx
                idx += 1
            iso.append(hmp[char]) 
        return iso
    
    return iso_group(s) == iso_group(t)


def lengthOfLastWord(s: str) -> int:
    # 58. Length of Last Word
    p = len(s) - 1

    while p >= 0 and s[p] == " ":
        p -= 1
    
    ans = 0

    while p >= 0 and s[p] != " ":
        ans += 1
        p -= 1
    return ans


def reverseStr(s: str, k: int) -> str:
    # 541. Reverse String II

    ans = []

    for i in range(0, len(s), 2*k):

        left = i
        right = min(i+k-1, len(s)-1)
        while right >= left:
            ans.append(s[right])
            right -= 1
        
        for i in range(i+k, min(i+2*k, len(s))):
            ans.append(s[i])
    
    return ''.join(ans)


def validWordAbbreviation(word: str, abbr: str) -> bool:
    # 408. Valid Word Abbreviation
    i = j = 0
    
    m, n = len(word), len(abbr)
    while i < m and j < n:
        skip = 0
        if word[i] == abbr[j]:
            i += 1
            j += 1
        elif abbr[j] == "0":
            return False
        elif abbr[j].isnumeric():
            while j < n and abbr[j].isnumeric():
                skip = int(abbr[j]) + skip*10
                j += 1
            i += int(skip)
        else:
            return False
    return i == m and j == n



def strStr(haystack: str, needle: str) -> int:
    # 28. Find the Index of the First Occurrence in a String
    # T: O(n*m)
    # S: O(1)

    m = len(haystack)
    n = len(needle)

    for window_start in range(m-n+1):
        for i in range(n):
            if needle[i] != haystack[window_start + i]:
                break
            
            if i == n - 1:
                return window_start
    return -1


def repeatedSubstringPattern(self, s: str) -> bool:
    # 459. Repeated Substring Pattern
    # s = p * k
    # s + s = p * k + p * k
    #       = head + p*(k-1) + p*(k-1) + tail

    return (s+s)[1:-1].find(s) != -1

def finalString(s: str) -> str:
    # 2810. Faulty Keyboard
    ans = deque()
    i_count = s.count("i") % 2 # 0: normal, 1: revesered

    for char in s:
        if char == 'i':
            i_count ^= 1
        else:
            if i_count:
                ans.appendleft(char)
            else:
                ans.append(char)
    return ''.join(ans)


def mostCommonWord(paragraph: str, banned: list[str]) -> str:
    # 819. Most Common Word
    # O(N+M) TS

    remove_punc = ''.join([char.lower() if char.isalnum() else ' ' for char in paragraph])
    remove_punc = remove_punc.split()

    banned_set = set(banned) 
    word_freq = dict()

    for word in remove_punc:
        if word not in banned_set:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    most_common_word = max(word_freq, key=word_freq.get)
    return most_common_word