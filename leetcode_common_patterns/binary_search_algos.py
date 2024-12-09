# arr is sorted --> binary search
# otherwise we can use heapq (n+k) * logk, O(k) space
def findClosestElements(arr: list[int], k: int, x: int) -> list[int]:
    """
    we are searching for a k + 1 (1 is outside k) windows 
    starting from left (mid):

    [arr[mid], ..., arr[mid + k]]
    ↑                  ↑
    起始點(包含)     結束點(不包含在k個數中)
    所以：

    當我們移動到 left = mid + 1 時，我們是確定當前的 mid 不可能是答案，因為我們已經看到了比它更遠的那個點（arr[mid + k]）
    當我們設置 right = mid 時，我們需要保留 mid 這個點，因為我們還沒有確定地看到它是不是最佳起始點
    數字正常分布之下，理論上left < x < right (也就是sliding window包含x)
    但如果某一遍數字特大 可能sliding window都在某一側

    """

    if len(arr) == k:
        return arr
        
    left = 0
    right = len(arr) - k
    
    while left < right:
        mid = left + (right - left) // 2
        
        # direction information
        if x - arr[mid] > arr[mid + k] - x:
            left = mid + 1
        else:
            right = mid
            
    return arr[left:left + k]

    
def searchMatrix(matrix: list[list[int]], target: int) -> bool:

    # key is to flatten all the indices
    # high = len(matrix) * len(matrix[0]) - 1
    # row = mid // cols
    # col = mid % cols

    low, high = 0, len(matrix) * len(matrix[0]) - 1
    cols = len(matrix[0])

    while low <= high:
        mid = low + (high - low) // 2
        row = mid // cols
        col = mid % cols
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] > target:
            high = mid - 1 
        else:
            low = mid + 1
    return False



