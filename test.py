import time
from typing import Callable


def timer(func: Callable) -> Callable:
	def wrapper(*args, **kwargs) -> None:
		start_time = time.time()
		func(*args, **kwargs)
		end_time = time.time()
		print(f"time elapsed: {end_time - start_time} s")
		return 
	return wrapper


@timer
def my_add(a: int, b: int) -> int:
	return a + b

if __name__ == "__main__":
	print(my_add(1, 2))

