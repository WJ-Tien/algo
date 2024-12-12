
def permute(nums):
    def backtrack(path, options):
        if len(path) == len(nums):  # 路徑長度達到目標
            result.append(path[:])
            return
        for i in range(len(options)):
            path.append(options[i])  # 選擇
            backtrack(path, options[:i] + options[i+1:])  # 選擇剩餘的
            path.pop()  # 撤銷選擇

    result = []
    backtrack([], nums)
    return result

def combine(n, k):
    def backtrack(start, path):
        if len(path) == k:  # 達到所需組合長度
            result.append(path[:])
            return
        for i in range(start, n + 1):  # 遍歷剩餘選項
            path.append(i)
            backtrack(i + 1, path)  # 避免重複選擇
            path.pop()  # 撤銷選擇

    result = []
    backtrack(1, [])
    return result

def subsets(nums):
    def backtrack(start, path):
        result.append(path[:])  # 每次都加入當前子集
        for i in range(start, len(nums)):
            path.append(nums[i])  # 選擇
            backtrack(i + 1, path)  # 遞歸下一步
            path.pop()  # 撤銷選擇

    result = []
    backtrack(0, [])
    return result
