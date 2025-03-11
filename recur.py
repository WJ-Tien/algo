# 123
# def decimal_to_binary(n: int):
# 	x = n

# 	ans = []
# 	while x:
# 		rem = x % 2
# 		ans.append(rem)
# 		x //= 2

# decimal_to_binary(233)

nums = [1, 3, 8, 12]
low, high = 0, len(nums) - 1
def bs(low, high, nums, target):
	while low <= high:
		mid = low + (high - low) // 2
		if nums[mid] == target:
			return mid + 1
		elif nums[mid] > target:
			high = mid - 1
		else:
			low = mid + 1
	return low

print(bs(low, high, nums, 10))
			


