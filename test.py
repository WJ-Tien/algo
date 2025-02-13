
def deprecated(reason="TEST"):
    def fancy(func):
        def wrapper(*args, **kwargs):
            print("I am a wrapper")
            func(*args, **kwargs)
        return wrapper
    return fancy

@deprecated
def exile(num):
    print(f"I am an exilar: {num}")
