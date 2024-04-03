class TrieNode:
    def __init__(self):
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