"""
前綴和（Prefix Sum）的概念：
「前綴和」是指從陣列的起始位置到目前位置的元素總和。
若兩個位置的前綴和分別為 prefix_sum[i] 和 prefix_sum[j] 並且 prefix_sum[j] - prefix_sum[i] == k，則說明從第 i+1 到第 j 的子陣列和為 k。
利用哈希表快速查找前綴和：
設計一個字典 prefix_count，記錄每個前綴和出現的次數。
當遍歷到某個位置時，計算 prefix_sum - k，如果該值已經存在於字典中，代表存在某個前綴和，可以構成和為 k 的子陣列。
    
    
"""

def subarraySum(nums: list[int], k: int) -> int:

    prefix_sum = 0
    prefix_count = {0: 1}
    ans = 0

    for num in nums:
        prefix_sum += num
        if prefix_sum - k in prefix_count:
            ans += prefix_count[prefix_sum - k]
        prefix_count[prefix_sum] = prefix_count.get(prefix_sum, 0) + 1
    return ans
    