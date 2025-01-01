
def decimal_to_binary(n: int):
	x = n

	ans = []
	while x:
		rem = x % 2
		ans.append(rem)
		x //= 2

decimal_to_binary(233)


