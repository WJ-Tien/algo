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
