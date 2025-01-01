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

def search(nums: list[int], target: int) -> int:
    # 33. Search in Rotated Sorted Array

    left, right = 0, len(nums) - 1

    while left <= right: 
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        
        # no need to include mid, 
        # since we check it at the beginning
        if nums[left] <= nums[mid]:
            # left part is sorted
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            # right part is sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1


def findMin(nums: list[int]) -> int:
    # 153. Find Minimum in Rotated Sorted Array

    left, right = 0, len(nums) - 1

    if nums[right] >= nums[left]:
        return nums[left]

    while left <= right:
        mid = left + (right -  left) // 2
        
        if nums[mid] > nums[mid + 1]:
            return nums[mid + 1]
        if nums[mid - 1] > nums[mid]:
            return nums[mid]
        if nums[mid] > nums[left]:
            left = mid + 1
        else:
            right = mid - 1


def findMedianSortedArrays(self, nums1: list[int], nums2: list[int]) -> float:
    # T: O(log(min(m+n)))
    # S: O(1)
    if len(nums1) > len(nums2):
        return self.findMedianSortedArrays(nums2, nums1)
    
    # search the shorter
    m, n = len(nums1), len(nums2)
    left, right = 0, m

    while left <= right:
        # very hard = =""
        # fixed amount of numbers should be in the left side
        # and in the right
        # we have partitionA in the left side
        # we still need (m+n+1)//2 - partitionA in the left side
        partitionA = left + (right - left) // 2
        partitionB = (m+n+1) // 2 - partitionA
        max_left_A = float("-inf") if partitionA == 0 else nums1[partitionA - 1]
        min_right_A = float("inf") if partitionA == m else nums1[partitionA]
        max_left_B = float("-inf") if partitionB == 0 else nums2[partitionB - 1]
        min_right_B = float("inf") if partitionB == n else nums2[partitionB]

        # all in left
        if max_left_A <= min_right_B and max_left_B <= min_right_A:
            if (m+n) % 2 == 0:
                return (max(max_left_A, max_left_B) + min(min_right_A, min_right_B))/2
            else:
                return max(max_left_A, max_left_B)
        elif max_left_A > min_right_B:
            right = partitionA - 1
        elif max_left_B > min_right_A:
            left = partitionA + 1
