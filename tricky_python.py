"""
Any module that contains a __path__ attribute is considered a package.
MRO 是 Method Resolution Order 的縮寫，字面上的意思是指 Python 在查找方法時候的尋找順序
Diamond problem --> C3 Linearization algo to solve --> finding next item in the MRO tuple

Override（覆寫）: 子類別重新定義父類別的方法	
多型（Polymorphism）: 不同類別的物件可以使用相同的介面（方法名稱），但有不同的行為
Override 是一種多型的實作方式，但多型不一定需要繼承

classmethod --> factory mode




"""
def test_args(*args, **kwargs):
	print(args)
	print(kwargs)
	for args in args:
		print(args)
	print("========")

	for kwarg, val in kwargs.items():
		print(kwarg, val)
	
 
# test_args(1,2,3, x=4, y=5, z=6)
# ===========================================================================================================

class test_class:
	order = 123
	def __init__(self):
		# self.order --> instance scope
		# test_class.order --> class scope --> shared by instances, even after init
		# if you use instance e.g., a.order = 888 --> won't effect other existing instances
		# and a.order will change this class variable to an instance variable
		# inheritance class also follow the rule
		# find instance --> if not found --> find class --> very special
		test_class.order += 10
	@classmethod
	def test2(cls):
		cls.order += 1
		return cls.order # shareable, as mentioned earlier

class B(test_class):
	def __init__(self):
		test_class.order += 100
	
# a = test_class()
# print(a.order)
# a.test2()
# print(a.__dict__) # {} --> indicate that at this moment, order is from the class variable.
# b = test_class()
# print(b.order)

# c = test_class()
# a.order = 888
# print(a.__dict__) # {"order": 888}
# print(a.order)
# print(b.order)
# print(c.order)
# print("==========")
# d = B()
# print(d.order)
# print(a.order)
# print(b.order)
# print(c.order)
# ===========================================================================================================

class human:
	# property has a conflict with __slots__

	__slots__ = ["name", "_age"] # inheritable

	def __init__(self, age: int) -> None:
		self._age = age

	@property
	def age(self) -> int:
		return self._age
	
	@age.setter
	def age(self, new_age: int) -> None:
		self._age = new_age

# s = human(100)
# print(s.age)
# s.age = 777
# print(s.age)

# ===========================================================================================================

# MRO example
class Animal:
    pass

class Mammal(Animal):
    pass

class Cat(Mammal):
    pass

Cat.mro() # Cat.__mro__
# (<class '__main__.Cat'>, <class '__main__.Mammal'>, <class '__main__.Animal'>, <class 'object'>)
# >>> Cat.__mro__
# (<class '__main__.Cat'>, <class '__main__.Mammal'>, <class '__main__.Animal'>, <class 'object'>)
# >>> Mammal.__mro__
# (<class '__main__.Mammal'>, <class '__main__.Animal'>, <class 'object'>)
# >>> Animal.__mro__
# (<class '__main__.Animal'>, <class 'object'>)

# ===========================================================================================================
class Animal:
    def walk(self):
        print("Animal is walking")

    def eat(self, food):
        print(f"{food} is yummy!")
	
class Cat(Animal):
	# override
	def walk(self):
		# Animal.eat(self, "罐罐")
		super().eat("罐罐")  # <-- 改用 super()
		print("Cat is walking")

# ===========================================================================================================
class Dog:
    def make_sound(self):
        print("Dog barks")

class Cat:
    def make_sound(self):
        print("Cat meows")

# 多型函式
def animal_sound(animal):
    animal.make_sound()  # 根據不同類別的物件，呼叫對應的方法

# dog = Dog()
# cat = Cat()

# animal_sound(dog)  # 輸出: Dog barks
# animal_sound(cat)  # 輸出: Cat meows