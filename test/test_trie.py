import pytest
from ..trie import Trie


def test_insert_and_search():
    trie = Trie()
    words = ["apple", "app", "banana", "ball", "cat"]
    
    # Insert words into the trie
    for word in words:
        trie.insert(word)
    
    # Test search for existing words
    assert trie.search("apple") == True
    assert trie.search("app") == True
    assert trie.search("banana") == True
    assert trie.search("ball") == True
    assert trie.search("cat") == True
    
    # Test search for non-existing words
    assert trie.search("ap") == False
    assert trie.search("ba") == False
    assert trie.search("apple1") == False

def test_startswith():
    trie = Trie()
    words = ["apple", "app", "banana", "ball", "cat"]
    
    # Insert words into the trie
    for word in words:
        trie.insert(word)
    
    # Test startsWith for existing prefixes
    assert trie.startsWith("app") == True
    assert trie.startsWith("ba") == True
    assert trie.startsWith("ca") == True
    
    # Test startsWith for non-existing prefixes
    assert trie.startsWith("ap1") == False
    assert trie.startsWith("c") == True
    assert trie.startsWith("d") == False

def test_empty_trie():
    trie = Trie()
    
    # Test search and startsWith on an empty trie
    assert trie.search("") == False
    assert trie.startsWith("") == True