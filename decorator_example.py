
def deprecated(reason="TEST"):
    def fancy(func):
        def wrapper(*args, **kwargs):
            print("I am a wrapper")
            print("Reason: ", reason)
            func(*args, **kwargs)
        return wrapper
    return fancy

@deprecated()
def exile(num):
    print(f"I am an exilar: {num}")

def countdown(n):
    while n > 0:
        yield n
        print(f"back:{n}")
        n -= 1

# c = countdown(10)
# next(c)
# next(c)

# base_kcal = 1600 
# print("Carbo: ", base_kcal * 0.60 / 4)
# print("Protein: ", base_kcal * 0.15 / 4)
# print("Fat: ", base_kcal * 0.25 / 9)
exile(10)
