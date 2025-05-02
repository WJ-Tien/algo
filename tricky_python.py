import unicodedata
from collections import Counter
from datetime import datetime, timedelta, timezone
from math import sqrt, log10 # 123

"""
python str split() --> white spaces --> " " or \n or \r or \t....

myobj 沒有定義 __bool__()
Python fallback 使用 __len__()
__len__() 回傳 0 ➜ False
__bool__ then __len__

// --> floor division (-3.1//1) --> -4
int --> toward zero int(-3.1) --> -3

dict key must be immutable/hashable --> otherwise it will raise unhashable type error
可雜湊物件（Hashable Object）從字面上看起來就是可以被雜湊函數所計算的物件，
在 Python 只要是不可變物件，例如整數、浮點數、字串、位元組，這些都是可以進行雜湊計算的；
相對的，如果是可變物件，像是串列、字典、集合，都是不可雜湊的。那麼 Tuple 呢？
這得看情況，Tuple 本身雖然是不可變物件，但還得看看裡面裝的元素是不是全部也都是可雜湊物件，如果裡面的元素都是可雜湊物件，
那麼這個 Tuple 就是可雜湊的。
在字典裡的 Key 有個規定，就是 Key 必須是可雜湊的物件，所以在字典裡的鍵不能是串列、字典、集合，但可以是 Tuple

Any module that contains a __path__ attribute is considered a package.
MRO 是 Method Resolution Order 的縮寫，字面上的意思是指 Python 在查找方法時候的尋找順序
Diamond problem --> C3 Linearization algo to solve --> finding next item in the MRO tuple

Override（覆寫）: 子類別重新定義父類別的方法	
多型（Polymorphism [ducktype]）: 不同類別的物件可以使用相同的介面（方法名稱），但有不同的行為
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

1. lookup data-descriptor
2. self.__dict__['d]
3. lookup non-data descriptor

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

__hash__ and __eq__ must be implemented at the same time --> 否則 set 和 dict 會出錯
✅ p1 和 p2 在 set 內被視為同一個物件，因為：
__eq__ 讓它們在比較時相等。
__hash__ 讓它們擁有相同的哈希值，確保 set 內不會重複存入。

def __eq__(self, other):
	return (self.id) == (other.id)

def __hash__(self):
	return hash(self.id)

generator物件同時也是一種iterable
for 迴圈或推導式的時候 Python 會自動幫我們搞定這個錯誤 (StopIteration)
Generator ad: prevent allocate too large memory block at once
Generator function: the function contains yield
generator iterator (generator object) 有實作 __iter__() 與 __next__() 2 個方法
generator iterator 是 iterator 無誤，所以可以當 iterator 使用！

iterable (not necessary an iterator, sort of like a container):
An object can be iterated over with for if it implements __iter__() or __getitem__() with sequence semantics.
iterable 就代表可以被 for 迴圈走訪！

__iter__ 讓物件可用 for ... in ... 迴圈遍歷。
__iter__() 方法規定必須回傳 1 個 iterator, 
所以用 iter() 函式將 [self.a, self.b, self.c] 轉成 1 個 iterator 。

iterator (iterable, stream of data):
其中選擇實作 __iter__() AND 實作 __next__() 方法的型態/類別，就稱為 iterator !
x = iter([1, 2, 3])

dict is iterable, but not an iterator (no __next__)

try error:
不管 try 區塊有沒有出錯，finally 區塊裡面的程式碼都會被執行。這個關鍵字通常用來做一些清理、善後的工作，例如關閉檔案、關閉資料庫連線等等，寫起來大概像這樣：
另一個關鍵字是 else，當 try 區塊沒有發生問題的時候，else 區塊裡面的程式碼才會被執行：

“Cell” objects are used to implement variables referenced by multiple scopes.


Decorator:
想要快速設計一個裝飾器，建議按照以下 3 步驟：
1️⃣ 先寫出原始函數（不加裝飾器）
2️⃣ 把想要的額外行為包裹起來（用 wrapper）
3️⃣ 用 @wraps(fn) 保持原函數資訊
✅ 3️⃣ 需要裝飾器帶參數？用「三層裝飾器」
如果 裝飾器本身需要參數，記得 先返回 decorator(fn)，再返回 wrapper：
# multiple decorators --> execute from inner to outer

「模組（Module）」是一個 Python 檔案，裡面可能會包含一些函數、變數和類別。
透過模組的設計，可以讓程式碼更有組織性，也更容易維護。
另外一個常會跟模組一起看到的名詞叫做「套件（Package）」，如果說剛才講到的模組是一個檔案的話，
那麼套件就是一個目錄、資料夾的概念。一個套件裡面可以放很多的模組，
或是放更多的子套件裡面再放更多的模組，
基本上就是個像檔案系統一樣的結構，一個目錄裡能放很多檔案以及更多的子目錄一樣

my_package/        # 套件目錄
│── __init__.py    # 讓這個目錄變成 Python 套件
│── math_utils.py  # 模組 1
│── string_utils.py # 模組 2

使用 __all__ 限制 import *
在 __init__.py 中：
__all__ = ["add"]
這樣 from my_package import * 只會導入 add()，而不會導入 reverse_string()。


__init__.py 可用來控制 當 import my_package 時導入哪些模組。
簡化導入：

例如，我們可以在 __init__.py 中加入：
from .math_utils import add
from .string_utils import reverse_string
這樣我們就能直接使用：
from my_package import add, reverse_string
而不需要：
from my_package.math_utils import add
from my_package.string_utils import reverse_string
--> 如果希望讓導入更簡潔，可以在 __init__.py 內部管理導入。(如上)


LEGB 规则（Local, Enclosing, Global, Built-in）：Python 查找变量时的顺序是： L –> E –> G –> B。
Local：当前函数的局部作用域。
Enclosing：包含当前函数的外部函数的作用域（如果有嵌套函数）。
Global：当前模块的全局作用域。
Built-in：Python 内置的作用域。

GIL: Global Interpreter Lock
當一個 Python 程式是以多線程 multithread 的方式在運行時，只有爭取到 GIL 的線程可以運行該程式。
每個線程基本上都在做這樣的事：
1. 爭取 GIL
2. 執行程式碼
3. 釋放 GIL

所以我們可以將 GIL 看作是一種「通行證」，只有取得通行證的線程才可以執行程式。
而一個直譯器只會有一個 GIL，因此，就算你把兩個不同的 thread 放在不同的核上，
程式在運行時依然只會有一個 thread 在運行。

在 CPython 中，會有一個內建的計數器或計時器，當達到一定的閥值時，就會強制釋放 GIL。 (or doing IO)
Python 會在執行一定數量的「字節碼指令」後，自動釋放 GIL。
或者，每 5ms Python 會嘗試強制釋放 GIL or reach 1000 bytecodes，
允許其他 Thread 執行（可透過 sys.setswitchinterval() 設定）。
但如果當前 Thread 還沒執行完，它仍然可能持有 GIL 更久！
Python 每 5ms 嘗試釋放 GIL，但 CPU 密集型 Thread 仍可能長時間持有 GIL
# GIL 只會對 CPU bound 任務有負面影響，對於 I/O bound 任務則沒有！
# 所以python 的threading如果是cpu bound的 他會一值傾向佔用該time slice
# --> python CPU-bound 的任務會傾向於一直持有 GIL (acquire lock release)，導致切換效率較低
# 而其他語言的情況是可以切換time slice (by OS)

Python = OS context switch + GIL 相關開銷
其他語言 = OS context switch

how to solve?
使用 multiprocess -> each process has its own interpreter --> no commu --> IPC (shared memory + message passing)
使用其他 Interpreter (Jython、IronPython 或 PyPy。)
等待 Python 官方移除 GIL (3.13)

如果 CPU 有多個核心，不同的 thread 可以真正並行執行（Parallel execution）-> distributed to different CPUs
如果 CPU 只有單核心，則多執行緒會透過 時間分片 來交錯執行（Concurrency）。
使用 std::mutex 保護共享資源，但頻繁使用會降低並行效能。

format string
f"{num}:.1f"

file read:
read()	讀取整個檔案（小型檔案）(w/ new line breaker)
readline()	逐行讀取
readlines()	讀取所有行（回傳 list）
with open(file_name, "r"):
	for line in f:
		# do something else

Python compiler
Python 虛擬機（Python Virtual Machine, PVM）逐行執行 Bytecode，轉換為 CPU 指令。
這就是 Python 比 C 慢的原因，因為它是 直譯執行，而不是預先編譯成機器碼！
frame --> bytecode (PVM[JIT, Cypython, PyPy], .pyc) --> CPU Instruction (Machine code)

Async/Await (same thread):
Coroutines are a more generalized form of subroutines. 
Subroutines are entered at one point and exited at another point. 
Coroutines can be entered, exited, and resumed at many different points. 
They can be implemented with the async def statement.
await 語法則是被用來告知 Python 可以在此處暫停執行 coroutine 轉而執行其他工作，
而且該語法只能在 coroutine 函式內使用，因此 async def 與 await 通常會一起出現。
另 1 個關於 await 語法的重點是 await 之後只能接 awaitables 物件，
例如 coroutine 或者是之後會介紹到的 Task, Future 以及有實作 __await__() 方法 的物件，
所以不是所有的物件或操作都能夠用 await 進行暫停。

coroutine 本身的特性有別於一般函式，一定要透過 event loop 排程後執行。
The event loop is the core of every asyncio application. 
Event loops run asynchronous tasks and callbacks, 
perform network IO operations, and run subprocesses.
event loop 關鍵運作部分是 1 個無窮迴圈（原始碼），
不斷地 loop 進行排程/執行非同步任務、回呼函式(callbacks)等工作

asyncio.to_thread() 可以將耗時執行的部分丟至 event loop 以外的另 1 個 thread 中執行，
每呼叫 1 次就會在 1 個新的 thread. --> to prevent the eventloop from blocking 

Tasks:
在 event loop 中，工作的執行是以 Task 為單位， 
event loop 一次僅會執行 1 個 Task, 如果某個 Task 正在等待執行結果，
也就是執行到 await 的地方，那麼 event loop 將會暫停(suspend)並將之進行排程，
接著切換執行其他 Task, 回呼函數(callback)或者執行某些 I/O 相關的操作。
Event loops use cooperative scheduling: an event loop runs one Task at a time. 
While a Task awaits for the completion of a Future, 
the event loop runs other Tasks, callbacks, or performs IO operations.

Future:
未來對象
本質：一個表示 尚未完成結果 的對象。
asyncio.Future() 是一個低階 API，通常不直接使用，而是讓 Task 來管理它。
Task 本質上是 Future 的子類，所以 Task 也是 Future。
主要用於讓某個操作（如 I/O）在未來某個時間點設定結果（set_result()）。

# asyncio.create_task() 回傳的 Task 並不需要等到使用 await 才會被執行，
Task 繼承自 Future, 因此 Future 是相對底層(low-level)的 awaitable Python 物件，
用以代表非同步操作的最終結果，一般並不需要自己創造 Future 物件進行操作，
多以 coroutine 與 Task 為主。

cls._instasnce = None
singleton: cls._instance = super().__new__(cls)


ULP: Unit in the last place = 2 ^ (exponent - 53)  --> 64 bits we have 53 for mantissa
big = 1e16 ~= 1 x 2^53
adjacent
small = 1.4
print(big + small - big)
np.spacing(1e16) = 2

mantissa bits 是浮點數尾數（mantissa）的長度：
float32 (單精度)：尾數有 23 位
float64 (雙精度)：尾數有 52 位

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
    #✅ 好處：
    # 節省記憶體（不再使用 __dict__）
    # 限制只能有 name, _age 屬性
    # 屬性 lookup 更快（底層是 array-like 結構）
    # ❌ 限制：
    # ❌ 不能動態新增屬性：

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

class A:
	print("class A")
class C(A): ...
class D(B, C):
    def method(self):
		# D.mro() == D->B->C->A
		# in this case self == D, start from C, so we finally get A
		# if super() --> start from D itself
        m = super(C, self).method() # print class A
        print(m)
        return 'Class D'
# 創建 Child 類別的實例
# child = Child("Alice", 10)
# print(child.name)
# print(child.age)
# ===========================================================================================================

class AgeValue:
	# obj: instance
	# obj_type: owner
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
        ''
        # print(f"{i} x {j} = {i * j}")

multiplication_table = [f"{i} x {j} = {i * j}" for j in range(1, 10) for i in range(1, 10)]
# 改用串列推導式來寫的話：
# print("\n".join(multiplication_table))

# ===========================================================================================================
# [*range(5)] = [0,1,2,3,4]
# *_
# combine array = [*arr1, *arr2]

# ===========================================================================================================

# dict unpacking
# >>> city = {"name": "台北", "population": 2600000}
# >>> location = {"lat": 25.04, "lng": 121.51}

# # 使用 ** 開箱字典再組合
# >>> info = {**city, **location}

# >>> info
# {'name': '台北', 'population': 2600000, 'lat': 25.04, 'lng': 121.51}
# 在合併字典的時候，不管是哪種合併方式，都需要注意如果有重複的 Key 的話，後面的值會蓋掉前面的值，也就是說 A 合併 B 跟 B 合併 A，結局可能是不一樣的。

# ===========================================================================================================
def create_counter():
    count = 0

    def inner():
        nonlocal count
        count += 1
        return count

    return inner

# 我在 create_counter() 函數裡設定了一個 count 變數，它會變成 inner() 函數的自由變數，
# 這裡也會發生閉包的行為。透過這個函數可以建立獨立的計數器，而且它們有各別的狀態：

counter1 = create_counter()
counter2 = create_counter()

# print(counter1())  # 印出 1
# print(counter1())  # 印出 2
# print(counter1())  # 印出 3

# print(counter2())  # 印出 1
# print(counter2())  # 印出 2

# print(counter1())  # 印出 4

# ===========================================================================================================

def log(level="INFO"):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            print(f"[{level}] 執行 {fn.__name__}()")
            return fn(*args, **kwargs)
        return wrapper
    return decorator

@log(level="WARNING")  # 帶參數的裝飾器
def my_function():
    print("這是我的函數")

# my_function()

# ===========================================================================================================

def deco1(func):
    def wrapper(*args, **kwargs):
        print("deco1 開始")
        result = func(*args, **kwargs)
        print("deco1 結束")
        return result
    return wrapper

def deco2(func):
    def wrapper(*args, **kwargs):
        print("deco2 開始")
        result = func(*args, **kwargs)
        print("deco2 結束")
        return result
    return wrapper

@deco1
@deco2
def my_cfunction():
    print("執行 my_function")

# my_cfunction()
# deco1 開始
# deco2 開始
# 執行 my_cfunction
# deco2 結束
# deco1 結束

# ===========================================================================================================

"""
結論
裝飾時（Wrapping）：

內層 (@decoB) 先執行，形成 decoB_wrapper。
外層 (@decoA) 再包住 decoB_wrapper，形成 decoA_wrapper。
執行時（Execution）：

先從 decoA_wrapper 開始，然後進入 decoB_wrapper，最後執行原始函數。
執行結束後，先從 decoB_wrapper 返回，再回到 decoA_wrapper。

裝飾器的"進入"是由外而內
實際函數執行是在最內層
裝飾器的"離開"是由內而外
 
"""

# Method	     Purpose	                            Input	         Output
# strftime()	Formats a datetime object as a string	datetime object	 Formatted string
# strptime()	Parses a string into a datetime object	Formatted string	datetime object

# string format time --> it's a string
# string parse time --> it's a datetime object
cur_time = datetime.now()
formatted = cur_time.strftime("%Y-%m-%d %H:%M:%S")
# print(formatted)
parsed = datetime.strptime(formatted, "%Y-%m-%d %H:%M:%S")
# print("Parsed Datetime:", parsed)
# print(type(parsed))
delta = timedelta(days=3)
# print(datetime.today() + delta)
# print(datetime.now().day)
# print(datetime.now().year)
# print(parsed.month)
naive_dt = datetime(2025, 2, 14, 12, 0, 0)
# Add UTC+5:30 timezone (e.g., India Standard Time)
ist = timezone(timedelta(hours=5, minutes=30))
aware_dt = naive_dt.replace(tzinfo=ist)
# print(ist) # UTC+05:30, default is None
# print(aware_dt) # 2025-02-14 12:00:00+05:30

# ===========================================================================================================

# the use of the key arg
def majorityElement(nums):
    counts = Counter(nums)
    return max(counts.keys(), key=counts.get)

def decimal_to_binary(n: int):
	x = n

	ans = []
	while x:
		rem = x % 2
		ans.append(str(rem))
		x //= 2
	return '0b' + ''.join(ans[::-1])


def decimal_to_hex(n: int):
    x = n
    hex_chars = '0123456789ABCDEF'  # 對應 0~15

    ans = []
    while x:
        rem = x % 16
        ans.append(hex_chars[rem])  # 把餘數轉成對應字元
        x //= 16
    return '0x' + ''.join(ans[::-1])

# print(decimal_to_binary(233))
# print(decimal_to_hex(233))
# print(bin(233))
# print(hex(233))

def is_prime(n: int) -> bool:
	if n < 2:
		return False
	
	for i in range(2, int(sqrt(n))+1):
		if n % i == 0:
			return False
	return True

def isPowerOfTwo(n: int) -> bool:
    # 1.
    x = n

    if x == 0:
        return False
    
    while n % 2 == 0:
        n //= 2
    return n == 1
    # 2. 
    if n == 0:
        return False
    return n & (n-1) == 0

    # 3.
    return n>=1 and log10(n)/log10(2)%1==0

def isPowerOfThree(self, n: int) -> bool:

    if n == 0:
        return False
    
    x = n
    while x % 3 == 0:
        x //= 3
    return x == 1

def hammingWeight(n: int) -> int:
    # 191. number of 1 bits

    ans = 0
    mask = 1
    for _ in range(32):
        if n & mask:
            ans += 1
        mask <<= 1
    return ans
# ===========================================================================================================

def countBits(n: int) -> list[int]:
    # 338. Counting Bits
    # P(x) = P(x/2) + x % 2
    # lsb = 0, x&1 == 0 == even
    # lsb = 1, x&1 == 1 == odd

    ans = [0] * (n+1)
    for i in range(n+1):
        ans[i] = ans[i >> 1] + (i % 2) # i & 1
    return ans


def friday_the_13th():
    ori_time = datetime.now()
    ans = ""
    for i in range(720):
        cur_time = ori_time + timedelta(days=i)
        if cur_time.weekday() == 4 and cur_time.day == 13:
            print(str(cur_time).split(" ")[0])
            ans = str(cur_time).split(" ")[0]
            break
    return ans



def remove_diacritics(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if not unicodedata.combining(c))

def is_anagram(left, right):
    left = remove_diacritics(left)
    right = remove_diacritics(right)
    hmp = dict()
    
    for char in left:
        if not char.isalpha():
            continue
        hmp[char.lower()] = hmp.get(char.lower(), 0) + 1
    
    for char in right:
        if not char.isalpha():
            continue
        if char.lower() not in hmp:
            return False
        
        hmp[char.lower()] -= 1
        if hmp[char.lower()] < 0:
            return False
    
    for freq in hmp.values():
        if freq > 0:
            return False
    return True
