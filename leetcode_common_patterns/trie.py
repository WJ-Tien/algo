# space: O(N * L * A)
# Time O(L)
# L: average length
# A: dict length (26, actually <= 26)
class TrieNode:
    def __init__(self):
        # children key = char
        #          value = TrieNode()
        self.children = {} 
        self.end_of_words = False

class Trie:

    def __init__(self):
        self._root = TrieNode()

    def insert(self, word: str) -> None:
        cur = self._root

        for char in word:
            if char not in cur.children:
                cur.children[char] = TrieNode() 
            cur = cur.children[char]
        cur.end_of_words = True

    def search(self, word: str) -> bool:
        cur = self._root
        for char in word:
            if char not in cur.children:
                return False
            cur = cur.children[char]
        return cur.end_of_words

    def startsWith(self, prefix: str) -> bool:
        cur = self._root
        for char in prefix:
            if char not in cur.children:
                return False
            cur = cur.children[char]
        return True


class WordDictionary:
    # leetcode 211

    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        # T: O(C^2 * L)
        # S: O(N*L + L[recursion]) 
        # N words, with average length L 
        # C .
        def dfs(start, node):
            for i in range(start, len(word)):
                if word[i] == ".":
                    # have to check all the children in node
                    for child in node.children.values():
                        if dfs(i + 1, child):
                            return True
                    return False
                else:
                    if word[i] not in node.children:
                        return False
                    node = node.children[word[i]]

            return node.is_end_of_word
        return dfs(0, self.root)