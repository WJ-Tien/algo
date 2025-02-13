"""
Any module that contains a __path__ attribute is considered a package.
MRO 是 Method Resolution Order 的縮寫，字面上的意思是指 Python 在查找方法時候的尋找順序
Diamond problem --> C3 Linearization algo to solve --> finding next item in the MRO tuple

Override（覆寫）: 子類別重新定義父類別的方法	
多型（Polymorphism）: 不同類別的物件可以使用相同的介面（方法名稱），但有不同的行為
Override 是一種多型的實作方式，但多型不一定需要繼承

classmethod --> factory mode

data descriptor (資料描述器)：
	同時實作 __get__ 和 __set__
	優先於實例的 __dict__
	可以攔截屬性的讀取與寫入
	確保屬性存取受控 (避免外部直接修改)
	支援資料驗證 (可以攔截並檢查賦值)
	實作唯讀屬性 (強制只讀取、不允許修改)
	確保屬性總是來自描述器，不會被覆蓋

non-data descriptor (非資料描述器)：
	只實作 __get__
	不影響 __dict__，允許實例屬性覆寫

__new__ | __init__ diff:
	__new__: 第一個參數是 cls（類別本身）
	__init__: 第一個參數是 self（實例本身）
	返回值差異:
	__new__: 必須返回一個物件實例
	__init__: 只能返回 None


print()、str()、format() 以及 F 字串會聽 __str__() 的。
repr() 這個函數會聽 __repr__() 的。

__str__: readable msgs
__repr__: devs debugging

The default implementation defined by the built-in type object calls object.__repr__().

迭代器一定是可迭代物件
but可迭代物件不一定是迭代器
迭代器協議的內容也很簡單，只要有實作 __iter__() 以及 __next__()

iterable:
An object can be iterated over with for if it implements __iter__() or __getitem__().
An object can function as an iterator if it implements next().

try error:
不管 try 區塊有沒有出錯，finally 區塊裡面的程式碼都會被執行。這個關鍵字通常用來做一些清理、善後的工作，例如關閉檔案、關閉資料庫連線等等，寫起來大概像這樣：
另一個關鍵字是 else，當 try 區塊沒有發生問題的時候，else 區塊裡面的程式碼才會被執行：

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
		# find instance --> if not found --> find class --> very special --> find __get__ (another class/instnace)
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
	# 會發現如果類別有設定 __slots__ 屬性的話，在建立物件的時候會把物件的 __dict__, __weak_ref__ 屬性給拿掉。

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

# ===========================================================================================================
class Parent:
    def __init__(self, name):
        print("Parent 初始化")
        self.name = name

class Child(Parent):
    def __init__(self, name, age):
        super().__init__(name)  # 調用父類的 __init__
        print("Child 初始化")
        self.age = age

# 創建 Child 類別的實例
# child = Child("Alice", 10)
# print(child.name)
# print(child.age)
# ===========================================================================================================

class AgeValue:
    def __get__(self, obj, obj_type):
        return 18

class Cat:
    age = AgeValue()

"""

self：這個 self 指的是 AgeValue 這個類別的實體，不是 kitty 這個實體喔！--> 描述器的實體
obj：承上，這個參數才是 kitty 實體。
obj_type：這個描述器掛在哪個類別裡，以上面的例子來說就是 Cat。

>>> kitty = Cat()
>>> kitty.__dict__
{}
>>> kitty.age
18

這是因為這個實體實作了 __get__() 方法，
當透過 . 的方式存取屬性或方法的時候，如果這個值剛好是某個類別的實體，
就會看看這個實體有沒有實作 __get__() 方法。如果有，Python 就會呼叫這個方法
如果描述器只有實作 __get__() 方法的時候，我們稱它為「非資料描述器」（Non-Data Descriptor）
它只能讀取屬性的值，但沒有寫入功能。0

find instance --> if not found --> find class --> not found --> find __get__ (another class/instnace)

# if we don't use __get__

class AgeValue:
    pass

>>> kitty = Cat()
>>> kitty.age
<__main__.AgeValue object>

"""

# ===========================================================================================================
class AgeValue:
    def __init__(self, age=0):
        self._age = age

	# Non-data descriptor
    def __get__(self, obj, obj_type):
        return self._age

	# Data Descriptor
    def __set__(self, obj, value):
        if value < 0 or value > 150:
            raise ValueError("年齡超過範圍")
        self._age = value
		# won't show in __dict__
		# 資料描述器會遮蔽（Shadow）物件的 __dict__，非資料描述器就沒這特性

class Cat:
    age = AgeValue()
# ===========================================================================================================

class Cat:
	age = AgeValue()

a = Cat()
b = Cat()
a.age = 90
# print(a.age, b.age)

class Tree:
	h = 100 # class variable

c = Tree()
d = Tree()
c.h = 777
print(d.h)
Tree.h = 800
print(d.h)
# find instance -> not found --> class
# __dict__ stores only instance variables -->
# which means, at the beginning, the {} is empty
# print(c.__dict__)  # {}  → 還沒有屬性，會使用類別屬性
# print(d.__dict__)  # {}  → 還沒有屬性，會使用類別屬性

# c.h = 777  # 這會在 c 實例內部新增 h
# print(c.__dict__)  # {'h': 777}  → c 有自己的 h，不再使用類別的 h
# print(d.__dict__)  # {}  → d 沒有 h，還是使用類別的 h

# Tree.h = 800  # 改變類別屬性
# print(d.h)  # 800，因為 d 還是使用類別屬性 h
# print(c.h)  # 777，因為 c 有自己的 h，不受類別影響
# 如果你希望修改 Tree.h 時，所有實例的 h 也都跟著改變，你應該避免讓實例擁有獨立的 h 屬性，可以使用 @property 來強制實例總是訪問類別屬性：


# ===========================================================================================================
class Parent:
    parent_attr = "parent"
    
    def parent_method(self):
        pass
        
    @classmethod
    def class_method(cls):
        pass

class Childs(Parent):
    child_attr = "child"
    
    def __init__(self):
        self.instance_attr = "instance"
    
    def child_method(self):
        pass
    
    @staticmethod
    def static_method():
        pass

c = Childs()
print(c.__dict__)  
# 只會輸出: {'instance_attr': 'instance'}

print(Child.__dict__)  
# 會包含: 'child_attr', 'child_method', 'static_method' 等
# 但不會包含繼承自 Parent 的屬性和方法

# instance call
# 會存在 __dict__ 的：
# 實例屬性 (instance attributes)
# 在類別中直接定義的一般方法

# class call
# 會存在 __dict__ 的：
# class attributes, classmethod, staticmethod, instance method (normal function la)

# ===========================================================================================================

class MyClass:
    def __new__(cls):
        # 這行實際上做了以下事情：
        # 1. super() 找到父類別 (object)
        # 2. 呼叫父類別的 __new__ 方法
        # 3. 把當前類別 (cls) 傳進去
        # 4. object.__new__ 會配置記憶體並建立 MyClass 的實例
        instance = super().__new__(cls)
        return instance

# 等同於：
# class MyClass:
#     def __new__(cls):
#         instance = object.__new__(cls)
#         return instance
# ===========================================================================================================
def hello():
    try:
        return "world"
    finally:
        return "kitty"
	# return kitty -> don't do this
# ===========================================================================================================

for i in range(1, 10):
    for j in range(1, 10):
        print(f"{i} x {j} = {i * j}")

multiplication_table = [f"{i} x {j} = {i * j}" for j in range(1, 10) for i in range(1, 10)]
# 改用串列推導式來寫的話：
# print("\n".join(multiplication_table))
# ===========================================================================================================