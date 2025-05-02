import pandas as pd
from typing import Callable

def add_on_f(p):
	def add_on(func: Callable) -> Callable:
		def wrapper(x, y):
			return f"--- {str(p) + str(y) + func(x, y)} ---"
		return wrapper
	return add_on

def add_on(func: Callable) -> Callable:
	def wrapper(x):
		return f"--- {'lower' + func(x)} ---"
	return wrapper
	
def add_on_2(func: Callable) -> Callable:
	def wrapper(x):
		return f"--- {'upper' + func(x)} ---"
	return wrapper


@add_on_f(777)
def check2(x, y):
	return str(x)

# @add_on_2
# @add_on
def check(x):
	return str(x) 
if __name__ == "__main__":
	# check = add_on(check)
	# check = add_on_2(check)
	# add_on_2(add_on(check))(2)
	# print(add_on(check)(2))
	# print(check(2))
	df = pd.DataFrame({"A":["A", "B", "A", "C", "C"], "B":[1,2,3,4,5]})
	print(df.groupby(['A']).mean().reset_index())
	


	
	
	
