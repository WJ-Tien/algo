"
Why Normal Form (NF)?
資料重複：同樣的資訊在不同地方出現多次。
更新困難：想改一個地方，結果要改好多個地方。
刪除或新增問題：一不小心就刪到或新增了不該放一起的資料。

1NF（第一正規化）：欄位只能存單一值，不能有「蘋果, 香蕉」這種多值情況。(atomic)
2NF（第二正規化）：非主鍵欄位要完全依賴主鍵，不能只依賴「部分」主鍵（例如「商品名稱」不應該依賴「訂單ID」）。
    -->如果該表有「複合主鍵」（也就是一張表的主鍵是由好幾個欄位一起組成），那麼每一個「非主鍵欄位」都必須依賴整個主鍵，而不是只依賴其中一部分。
3NF（第三正規化）：非主鍵欄位不能透過其他非主鍵欄位來獲得（例如「客戶名稱」應該放到客戶表，而不是訂單表）。

第二正規化 (2NF) - 消除「部分依賴」，避免重複
在 1NF 之後，現在看看這張表：

訂單ID	客戶名稱 商品ID	商品名稱
001	    小明	A01	   蘋果
001	    小明	A02    香蕉
002	    小華	A03	   西瓜
問題：

主鍵是「訂單ID + 商品ID」的組合（因為一張訂單可能包含多個商品）。
但是「商品名稱」只依賴於「商品ID」，與「訂單ID」無關，這是部分依賴。
例如：「A01（蘋果）」在多個訂單裡都是蘋果，沒必要重複存「蘋果」。
解決方法：拆成兩張表：

訂單-商品表：

訂單ID	客戶名稱	商品ID
001	小明	A01
001	小明	A02
002	小華	A03
商品表：

商品ID	商品名稱
A01	蘋果
A02	香蕉
A03	西瓜
✅ 這樣就符合 2NF，因為「商品名稱」現在只依賴於「商品ID」，不會跟「訂單ID」混在一起。

第三正規化 (3NF) - 消除「傳遞依賴」，避免間接關係
現在看看這張表：

訂單ID	客戶ID	客戶名稱	客戶地址
001	C01	小明	台北
002	C02	小華	高雄
問題：

「客戶名稱」和「客戶地址」是依賴於「客戶ID」的，而不是直接依賴「訂單ID」。
也就是：「訂單ID → 客戶ID → 客戶名稱、客戶地址」，這是「傳遞依賴」。
解決方法：拆成兩張表：

訂單表：

訂單ID	客戶ID
001	C01
002	C02
客戶表：

客戶ID	客戶名稱	客戶地址
C01	小明	台北
C02	小華	高雄
✅ 這樣就符合 3NF，因為「客戶名稱」和「客戶地址」直接依賴「客戶ID」，而不是透過「訂單ID」來間接關聯。


2NF：避免同樣的資料一直重複寫（像是顧客名字）
3NF：避免改一個資料要改很多處（像是供應商電話）
-------------------------------------------------------------------------------------------------

BCNF：任何能決定其他欄位的條件 (決定元, Determinant) ，都必須是某個「候選鍵」(Candidate Key)。
換句話說，如果一個欄位 (或欄位組合) 可以決定表中的其他欄位，那這個欄位(或欄位組合) 本身就要能唯一識別整筆資料。
BCNF 其實是 3NF 的「更嚴格版本」，有些在 3NF 不被認為是問題的依賴關係，在 BCNF 裡就會被挑出來。
範例：課程、老師與教室
情境設定
一位老師可以教多門課程。
但每位老師在校內有一間「專屬」的固定教室。
一門課程可以由多位老師共同教授（或輪流上課）。
表格設計（違反 BCNF 的狀況）
我們設計一個表 ClassSchedule，用來記錄「課程、老師、教室」的對應關係：

課程ID	老師ID	教室ID
C001	T01	R101
C002	T01	R101
C001	T02	R202
C003	T03	R303
...	...	...
主鍵 (Primary Key)：

假設這張表設定 (課程ID, 老師ID) 作為複合主鍵
表示同一門課程可以對應到多位老師，也可以多筆紀錄
功能相依 (Functional Dependencies)：
(課程ID,老師ID) → 教室ID
（因為主鍵可以決定教室是哪一間，這是表面上看起來符合 3NF 的原因）
但還有： 老師ID → 教室ID
（在這個情境下，每位老師都固定一間教室，也就是只要知道「老師ID」，就能知道「教室ID」是哪一間）
為什麼 3NF 可能「表面上」覺得沒問題？
只看前兩點：「非主鍵（教室ID）依賴主鍵（課程ID+老師ID）」，感覺沒有依賴在其他非主鍵欄位之上，所以很多人會覺得「好像已經 3NF 了」。
為什麼違反 BCNF？
在 BCNF 中，任何能決定其他欄位的欄位（決定元），必須是候選鍵。
在這個例子裡，「老師ID」就能決定「教室ID」。但是「老師ID」並不是表的候選鍵（因為一位老師可教多門課程，單憑 老師ID 無法唯一分辨表裡的每一筆紀錄）。
也就是：
老師ID → 教室ID
但 老師ID 不是候選鍵
這樣就違反了 BCNF 的要求。

--------------------------------------------------------------------------------------------
1. 主鍵 (Primary Key): not null, unique(), minimal, one primary key
2. 候選鍵（Candidate Key）指的是在資料表中，一組（或一個）欄位能唯一辨識一筆紀錄，而且這組欄位不能再縮小（也就是說，如果拿掉任何一個欄位，就不再能唯一辨識資料）。
    以下幾個重點讓你更容易理解：
    能唯一辨識：
    只要你知道「候選鍵」欄位的值，就可以在表裡精準找到對應的那一筆或那幾筆資料（通常是一筆）。
    例如：
    學生ID, 課程ID
    學生ID,課程ID 可能一起可以辨識一筆「選課」資料。若只用「學生ID」或只用「課程ID」，可能會對應到多筆資料，就不夠唯一。
    最小性 (Minimality)：
    這組欄位合在一起「獨一無二」的同時，不能再拿掉任何欄位，否則就失去唯一辨識的能力。
    如果可以拿掉其中一欄，還能唯一辨識，就代表原來那組欄位不是最小的組合，那它就不是「候選鍵」。

    在一張表中，可能有不只一個候選鍵（多組欄位都能唯一辨識資料）。
    其中一個被選定當作「主鍵」（Primary Key），其他的都還是「候選鍵」，只是沒被選來當主鍵而已。

3. Super Key（超鍵）
    定義：

    在一張資料表裡，一個或多個欄位的組合，只要能「唯一識別 (Identify)」每一筆資料，就是一個 Super Key。
    但 Super Key 不要求最小化，也就是說，它可以包含多餘的欄位，只要這個組合仍然能分辨每筆資料就算是 Super Key。
    例子：

    假設在「學生」表裡，有欄位：(學號, 姓名, 身分證字號, 手機號碼)。
    「(學號, 姓名)」可以唯一識別學生嗎？
    如果光是「學號」就已經唯一了，加上「姓名」也還是能唯一識別，但這樣就多餘了（因為其實只靠「學號」就已經可以分辨每筆資料）。
    所以「(學號, 姓名)」是一個 Super Key，但不是最小組合。

4. Alternate Key（替代鍵） (剩下的沒有被選為 Primary Key 的 Candidate Key。)
   是指「除了主鍵（Primary Key）以外，其它可用來唯一識別資料的 Candidate Key」。
   定義：

    是指「除了主鍵（Primary Key）以外，其它可用來唯一識別資料的 Candidate Key」。
    通常我們會選擇「學號」當主鍵，或選擇「員工編號」當主鍵，那麼系統中如果還有「身分證字號」、「護照號碼」等也能唯一辨識的欄位，就屬於 Alternate Key。
    例子：

    一張「學生」表裡，「學號」被選為 Primary Key，那「身分證字號」如果也能唯一，就屬於 Alternate Key。

    Candidate Key 和 Alternate Key 其實是一個「包含」與「被剩下」的關係：
    所有 Alternate Key 都是 Candidate Key，
    但不是所有 Candidate Key 都會變成 Alternate Key，因為其中有一個被選去當 Primary Key 了。

5. Composite Key (複合鍵)
    定義：

    指的是「用兩個或以上的欄位組合在一起」才足以構成「唯一識別」的鍵。
    當單一欄位無法保證唯一，但多個欄位合起來就能唯一，這組合就叫做 Composite Key。
    例子：

    在「選課」(Enrollment) 表裡面，可能有（學生ID、課程ID）一起做主鍵，單獨「學生ID」或「課程ID」都不唯一，但是合起來就能唯一識別一筆「這位學生選了哪堂課」的紀錄。
    在多對多關係的「關聯表」(Junction Table) 中很常見。

6. Foreign Key（外鍵）
    定義：

    在一張表裡面，用來參照(Refernce)另一張表主鍵(或唯一的 Candidate Key)的欄位(或欄位組合)。
    外鍵會「連到」另一張表的「Primary Key 或 Candidate Key」，用來表明兩張表之間的關係(一對多、多對多等)。
    為什麼需要：

    外鍵可以保證資料的一致性(Referential Integrity)，如果外鍵指的那筆記錄在父表(Parent Table)不存在，就不能新增這筆資料，或是如果父表刪除/更新了那筆資料，也會影響或限制外鍵表中的資料。
    例子：

    在「訂單明細」表中，CustomerID 可能是指向「顧客」表(父表)的 CustomerID (PK)；
    在「選課」表中，學生ID 和 課程ID 都會是外鍵，分別連到「學生」表和「課程」表的主鍵。
--------------------------------------------------------------------------------------------

normal func
agg func
window func

-- CTE (Common Table Expressions) 通用資料表運算式 --> with tmp as (sub_query)

agg func --> NULL WON'T be included, but count will (count(*))
primary: unique + non-null
NULL 不會參與比較 a > 10, if a has any null values --> these won't be select or oped.
NULL 表示的是什麼都沒有，它與空字串 ('')、數字 0 並不等價，且不能用於比較！
例如：<expr> = NULL 或 NULL = '' 的結果為 FALSE。
要判斷 NULL，必須使用 IS NULL 或 IS NOT NULL 來進行檢查。

comparison with null value it won't give true or false values
bonus IS NULL 這個條件是 必要的，
因為在 SQL 中，NULL 不是數字，也不能用來比較大小，所以 b.bonus < 1000 不會包含 NULL 值。

self join --> need on (emp vs mgr)
cross join --> A rows * B rows (different table)
full (outer) join -> all rows, no matter matched or not.

INSERT INTO --> need both tables exist
SELECT INTO --> create a new table from the existing one

rank: 1 1 3
dense_rank: 1 1 2 2 3
row_number: 1 2 3 4 
over (partition something orderby another)
--> rank, dense_rank, row_number, range, lead, lag
diff: with groupby --> groupby reduce the #of cols while over-partition keeps
over means do ops on a dataset
lag/lead 是offset row (default respect NULLs)

limit offset --> offset first then limit
e.g., limit 3 offset 2 --> first offset 2 rows, and choose the last three in a row. 

view -> virutual table --> select only -> update when querying 
materialized view (cache like) -> with real data -> need to update manually
store procedure (SP): sort of like function in SQL, can reduce IO/CPU workload
# IN(read), OUT(return the output), INOUT (best, bidirection)
Common Table Expression (CTE): temporary table (with temp_name as)

ACID
Atomic: all successful or all failed
Consistency: e.g., bank balance must >= 0
isolation: each transaction is independent from each other (e.g., prevent phantom read)
durability: store the data forever 

INDEX: B+ index (data distriubtion), 重複越少效率越好 速度越快 (balanced/non-balanced tree)
Partition (same table, e.g. hash partition)
Sharding (tables in different physical storage)
Clustering: store similar data in the neighboring physical address (e.g., disk)

✅ 最佳實踐
🔹 何時適合加索引？
唯一值較多（高選擇性），如 id, email。
經常出現在 WHERE 條件中且能顯著過濾數據，如 order_date。
經常用於 JOIN 或 GROUP BY，如 customer_id。
🔹 何時不適合加索引？
大量重複值（低選擇性），如 gender, status。
表很小（< 1000 行），全表掃描更快。
頻繁 INSERT / UPDATE，導致索引維護成本高。



DELETE：刪除特定記錄（可回滾）
📌 逐行刪除表中的資料，可以加 WHERE 條件
📌 可回滾 (ROLLBACK)，因為會記錄到 UNDO LOG
📌 會觸發 DELETE 觸發器 (Trigger)
📌 執行速度較慢，因為它需要記錄每一行的刪除

TRUNCATE：清空表（不可回滾）
📌 刪除整個表的所有資料，但不刪表結構
📌 不可回滾 (ROLLBACK)，因為不會記錄 UNDO LOG
📌 不會觸發 DELETE 觸發器
📌 執行速度比 DELETE 快，因為它直接清空表

DROP：刪除表
📌 刪除整個表，包括結構、索引、約束
📌 不可回滾 (ROLLBACK)，因為會直接刪除表
📌 刪除後，表無法恢復，需要重新 CREATE TABLE
📌 執行速度最快

index的類別分為 B-tree 與 Hash 2 種，這 2 種有各自適合的情境，譬如某些不重複的欄位，就適合使用 Hash 作為索引，不過 Hash 索引無法進行範圍查詢和排序，因此要考慮清楚
partition: same table's order (hash, range ....)
clustering: physical storage order (i.e. disk)


SELECT * 
FROM Delivery 
WHERE (customer_id, order_date) IN (
    (1, '2024-01-01'),
    (2, '2024-01-02'),
    (3, '2024-01-03')
);
這裡的 IN 作用於多個欄位，會匹配 (customer_id, order_date) 是否與提供的數組（tuples）相符。


標準 SQL 中，當你在 GROUP BY 某些欄位（可以是一個或多個「鍵」）時，SELECT 子句裡 只能出現：
與 GROUP BY 條件中一模一樣的欄位（或同等於這些欄位的表達式），以及
聚合函數（SUM, COUNT, MIN, MAX, AVG...）的結果。
任何「沒有在 GROUP BY 出現、也沒有被聚合」的欄位，都會引發 SQL 錯誤（產生不確定的結果）。
select g_col, max(a), min(b) --> OK
select g_col, max(a) --> OK
select g_col, max(a), c --> WRONG
# select g_col_1, g_col_2, MIN(price) AS min_price,, MAX(price) AS max_price -> OK
from table
group by g_col
# group by g_col_1, g_col_2 -> OK (ref #)
# You can put expression in the sum e.g., sum(price > 0)
select g_col --> (OK)
group by g_col


<窗口函數> OVER (
    PARTITION BY <分組欄位>
    ORDER BY <排序欄位>
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
)
🔹 ROWS BETWEEN 的定義
6 PRECEDING：表示當前行往上數 6 行（包含這 6 行）。
CURRENT ROW：表示當前行。
這樣的範圍就是「當前行 + 前 6 行」，共 7 行。

ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW	從第一行到當前行的累積計算（累積總和）。
ROWS BETWEEN 6 PRECEDING AND CURRENT ROW	  計算當前行 + 前 6 行（移動平均）。
ROWS BETWEEN CURRENT ROW AND 6 FOLLOWING	計算當前行 + 後 6 行（未來 7 天平均）。
ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING	計算整個表的聚合值（如 AVG() 計算全表平均）。

Coalesce: return first non-null value; if all are nulls, then return ull



"
--197. Rising Temperature
-- select w1.id from weather w1, weather w2 --> return all combs n^2 <- self-join
-- By doing a self-join on the Weather table, we create a Cartesian product of the table with itself, creating pairs of days
-- Postgresql
select w1.id from weather w1, weather w2
where w1.temperature > w2.temperature
and w1.recordDate::date - w2.recordDate::date = 1

-- MYSQL - 1
WITH PreviousWeatherData AS
(
    SELECT 
        id,
        recordDate,
        temperature, 
        LAG(temperature, 1) OVER (ORDER BY recordDate) AS PreviousTemperature,
        LAG(recordDate, 1) OVER (ORDER BY recordDate) AS PreviousRecordDate
    FROM 
        Weather
)
SELECT 
    id 
FROM 
    PreviousWeatherData
WHERE 
    temperature > PreviousTemperature
AND 
recordDate = DATE_ADD(PreviousRecordDate, INTERVAL 1 DAY);
-- and DATEDIFF(recordDate, previousRecordDate) = 1

-- MySQL - 2
SELECT w1.id FROM Weather w1
JOIN Weather w2
ON DATEDIFF(w1.recordDate, w2.recordDate) = 1
WHERE w1.temperature > w2.temperature


-- 1661. Average Time of Process per Machine
SELECT 
    machine_id,
    ROUND(SUM(CASE 
                  WHEN activity_type = 'start' THEN -timestamp 
                  ELSE timestamp 
              END) * 1.0
          / COUNT(DISTINCT process_id), 3) AS processing_time
FROM 
    Activity
GROUP BY 
    machine_id;


-- 620. Not Boring Movies
select * from Cinema
where id % 2 = 1 
and description <> "boring"
order by rating des

-- 1978. Employees Whose Manager Left the Company
-- manager is also an employee 
select employee_id from Employees
where salary < 30000
and manager_id NOT IN (select employee_id from Employees)
order by employee_id


-- 1303. Find the Team Size
-- using window function
select employee_id, count(team_id) over (partition by team_id) as team_size 
from Employee


-- 2356. Number of Unique Subjects Taught by Each Teacher
select teacher_id, count(distinct subject_id) as cnt from Teacher
group by teacher_id


-- 2989. Class Performance
with tmp as (
    select (assignment1 + assignment2 + assignment3) as total_score
    from Scores
)
select (max(total_score) - min(total_score)) as difference_in_score from tmp


-- 3338. Second Highest Salary II
with tmp as (
    select emp_id, dept, dense_rank() over (partition by dept order by salary desc) as rnk 
    from employees
)
select emp_id, dept from tmp
where rnk = 2
order by emp_id


-- 2339. All the Matches of the League
select t1.team_name as home_team, t2.team_name as away_team
from Teams as t1
cross join Teams as t2
where (on) t1.team_name != t2.team_name


-- 2985. Calculate Compressed Mean
select round(sum(item_count * order_occurrences) / sum(order_occurrences), 2) as average_items_per_order
from Orders 


-- 1571. Warehouse Manager
with tmp as (
    select product_id, (Width*Length*Height) as volume
    from Products
)
select w.name as warehouse_name, sum(p.volume * w.units) as volume
from Warehouse as w
inner join tmp as p
on w.product_id = p.product_id
group by w.name


-- 2084. Drop Type 1 Orders for Customers With Type 0 Orders
select order_id, customer_id, order_type 
from Orders
where order_type = 0
or (order_type = 1 and customer_id not in (select customer_id from orders where order_type = 0))


-- 3150. Invalid Tweets II
select tweet_id from Tweets
where length(content) > 140
or length(content) - length(replace(content, "#", '')) > 3 
or length(content) - length(replace(content, "@", '')) > 3 


-- 1308. Running Total for Different Genders
-- running total
-- sum over --> running total
select gender, day,
sum(score_points) over(partition by gender order by day) as total
from Scores


-- 1445. Apples & Oranges
-- solved
select sale_date, SUM(CASE WHEN fruit = 'apples' THEN sold_num ELSE -sold_num END) as diff
from Sales
group by sale_date


-- 1795. Rearrange Products Table
-- pivot using union 

Products table:
+------------+--------+--------+--------+
| product_id | store1 | store2 | store3 |
+------------+--------+--------+--------+
| 0          | 95     | 100    | 105    |
| 1          | 70     | null   | 80     |
+------------+--------+--------+--------+
Output: 
+------------+--------+-------+
| product_id | store  | price |
+------------+--------+-------+
| 0          | store1 | 95    |
| 0          | store2 | 100   |
| 0          | store3 | 105   |
| 1          | store1 | 70    |
| 1          | store3 | 80    |
+------------+--------+-------+
select product_id, 'store1' as store, store1 as price
from products
where store1 is not NULL
union
select product_id, 'store2' as store, store2 as price
from products
where store2 is not NULL
union
select product_id, 'store3' as store, store3 as price
from products
where store3 is not NULL


-- 1853. Convert Date Format
SELECT DATE_FORMAT(day, "%W, %M %e, %Y") AS day FROM Days;


-- 1393. Capital Gain/Loss
select stock_name, sum(CASE WHEN operation = 'Buy' THEN -price ELSE price END) as capital_gain_loss


-- 1581. Customer Who Visited but Did Not Make Any Transactions
select v.customer_id, sum(CASE WHEN t.transaction_id is NULL THEN 1 ELSE 0 END) as count_no_trans 
from Visits as v
left join Transactions as t
on v.visit_id = t.visit_id
group by v.customer_id
having count_no_trans > 0


-- 1280. Students and Examinations
-- https://www.fooish.com/sql/cross-join.html
WITH StudentSubjects AS (
    -- 1. 產生所有學生 × 所有科目
    SELECT s.student_id, s.student_name, sub.subject_name
    FROM Students s
    CROSS JOIN Subjects sub
)

SELECT 
    ss.student_id,
    ss.student_name,
    ss.subject_name,
    COUNT(e.subject_name) AS attended_exams
-- count (e.) is critical, e has NULL, but ss does not
FROM StudentSubjects ss
LEFT JOIN Examinations e 
ON ss.student_id = e.student_id 
AND ss.subject_name = e.subject_name
GROUP BY ss.student_id, ss.student_name, ss.subject_name
ORDER BY ss.student_id, ss.subject_name;

-- better solution
-- two CTEs
with all_stu_sub as (
    select * from Students
    cross join Subjects
),

exam_grp as (
    select student_id, subject_name, COUNT(*) as 'attended_exams' from Examinations
    group by student_id, subject_name
)

select a.student_id, a.student_name, a.subject_name, COALESCE(e.attended_exams, 0) as 'attended_exams'
from all_stu_sub as a
left join exam_grp as e
on a.student_id = e.student_id
and a.subject_name = e.subject_name
order by a.student_id, a.subject_nam


-- 1934. Confirmation Rate
with tmp as (
    select s.user_id, c.action from Signups as s
    left join Confirmations as c
    on s.user_id = c.user_id
)

SELECT 
    user_id, 
    ROUND(
        SUM(CASE WHEN action = 'confirmed' THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 
        2
    ) AS confirmation_rate 
FROM tmp 
GROUP BY user_id;


-- 1251. Average Selling Price
select p.product_id, COALESCE(ROUND(SUM(p.price * u.units) / SUM(u.units), 2), 0) as average_price
from Prices as p
left join UnitsSold as u
on p.product_id = u.product_id
and purchase_date between start_date and end_date 
group by p.product_id


-- 1075. Project Employees I
select p.project_id, ROUND(AVG(e.experience_years), 2) as average_years
from Project as p
left join Employee as e
on p.employee_id = e.employee_id
group by p.project_id


-- 1633. Percentage of Users Attended a Contest
with tmp as (
    select count(*) as total_number from users
)
select r.contest_id, ROUND((count(u.user_id) *100 / (select total_number from tmp)), 2) as percentage from Users as u
inner join Register as r
on u.user_id = r.user_id
group by r.contest_id
order by percentage desc,  r.contest_id as


-- 1211. Queries Quality and Percentage
-- remember casting --> mutiply by .0
-- the use of avg
-- poor quality probably need coalesce for 0
select query_name,
round(avg(rating*1.0 / position), 2) as quality,
round(sum(case when rating < 3 then 1 else 0 end) * 100.0 / count(query_name), 2) as poor_query_percentage
from Queries
group by query_name

--  1193. Monthly Transactions I
SELECT DATE_FORMAT(trans_date, '%Y-%m') AS month, country,
COUNT(state) as trans_count,
SUM(CASE WHEN state = 'approved' THEN 1 ELSE 0 END) as approved_count,
SUM(amount) as trans_total_amount,
SUM(CASE WHEN state = 'approved' THEN amount ELSE 0 END) as approved_total_amount 
from Transactions
group by month, country


-- 1174. Immediate Food Delivery II
with tmp1 as (
    select customer_id, order_date, dense_rank() over (partition by customer_id order by order_date) as rnk
    from Delivery
),
tmp2 as (
    select d.customer_id, d.order_date, (CASE WHEN d.order_date = d.customer_pref_delivery_date THEN 'immediate' ELSE 'scheduled' END) as type 
    from Delivery as d
    inner join tmp1 as t
    on t.customer_id = d.customer_id
    and t.order_date = d.order_date
    where rnk = 1
)
select ROUND(AVG(type = 'immediate') * 100.0, 2) as immediate_percentage from tmp2

-- 1174. Immediate Food Delivery II
SELECT 
    ROUND(AVG(order_date = customer_pref_delivery_date) * 100.0, 2) AS immediate_percentage
FROM Delivery
WHERE (customer_id, order_date) IN (
    SELECT customer_id, MIN(order_date) AS first_order_date
    FROM Delivery
    GROUP BY customer_id
)


-- 550. Game Play Analysis IV
-- read the problem statement properly
WITH first_login_cte AS (
    -- 每位玩家的「第一次登入」日期
    SELECT 
        player_id, 
        MIN(event_date) AS first_login
    FROM Activity
    GROUP BY player_id
),
logged_again_cte AS (
    -- 只看「首登日 + 1 天」是否有登入
    SELECT DISTINCT f.player_id
    FROM first_login_cte f
    INNER JOIN Activity a
        ON a.player_id = f.player_id
       AND DATEDIFF(a.event_date, f.first_login) = 1
)
SELECT 
    ROUND(
        COUNT(DISTINCT lac.player_id) *1.0
        / (SELECT COUNT(DISTINCT player_id) FROM Activity)
    , 2
    ) AS fraction
FROM logged_again_cte as lac


-- 1141. User Activity for the Past 30 Days I
-- INTERVAL 29 DAY --> D-30
-- INTERVAL N DAY --> D-(N+1)
select activity_date as "day", count(distinct user_id) as active_users
from Activity
group by activity_date
having activity_date between DATE_SUB("2019-07-27", INTERVAL 29 DAY) and "2019-07-27"


-- 1070. Product Sales Analysis III
-- agg func, one main col and an agg col
-- (a, b) in subquery
with tmp as (
    select product_id, min(year) as first_year
    from Sales
    group by product_id
)
select product_id, year as first_year, quantity, price
from Sales
where (product_id, year) in (select * from tmp)


-- 1045. Customers Who Bought All Products
-- read the problem properly (pk and fk)
select customer_id
from Customer
group by customer_id
having count(distinct product_key) = (select count(*) from Product)


-- 610. Triangle Judgement
select x, y, z, (CASE WHEN x + y > z and y + z > x and x + z > y THEN "Yes" ELSE "No" END) as "triangle"
from Triangle

-- 610. Triangle Judgement
select x, y, z, if (x + y > z and y + z > x and x + z > y, "Yes", "No") as 'triangle'
from triangle


-- 1789. Primary Department for Each Employee
-- subquery
-- first select 'N' with only one row
-- and finally selecet "Y"
with tmp as (
    select employee_id, COUNT(distinct department_id) as cnt
    from Employee
    group by employee_id
    having cnt = 1
)
select employee_id, department_id
from Employee
where primary_flag = "Y"
or employee_id in (select employee_id from tmp)


-- 1731. The Number of Employees Which Report to Each Employee
-- every employee could be a mgr of any other
with mgr as (
    select employee_id, name from Employees
)
select m.employee_id, m.name, 
count(e.reports_to) as reports_count, 
ROUND(avg(e.age)) as average_age
from mgr as m
inner join Employees as e
on m.employee_id = e.reports_to
group by m.employee_id, m.name
order by employee_id


-- 626. Exchange Seats
-- a smart way to swap 
select (CASE WHEN id % 2 = 0 THEN id -1 
             WHEN id % 2 = 1 and id < (select count(*) from Seat) THEN id + 1
             ELSE id END) as `id` -- edge case
             , student from Seat
order by `id`


-- 180. Consecutive Numbers
with tmp as (
    select num, lag(num) over (order by id) as prev, lead(num) over (order by id) as next
    from Logs
)
select distinct num as ConsecutiveNums from tmp
where num = prev and num = nex


-- 1667. Fix Names in a Table
-- substr (str, pos, len)
-- start from 1 (inclusive) and proceed 1, so which is itself
-- 2, None --> start from 2 (inclusive) til the end
select user_id, CONCAT(upper(substr(name, 1, 1)), lower(substr(name, 2))) as name from Users
order by user_id


-- 1327. List the Products Ordered in a Period
select p.product_name, SUM(o.unit) as `unit`
from Products as p
inner join Orders as o
on p.product_id = o.product_id
where DATE_FORMAT(order_date, '%Y-%m') = '2020-02'
-- where left(order_date, 7) = '2020-02
group by p.product_id
having `unit` >= 10


-- 196. Delete Duplicate Emails
-- 在許多資料庫系統（特別是 MySQL）中，如果你在一個 DELETE（或 UPDATE）指令裡，同時又想從「同一張表」進行 SELECT 以取得條件，往往會遇到以下錯誤或限制：
-- You can't specify target table 'XXX' for update in FROM clause
-- 也就是說，不允許直接在 DELETE FROM Person 的同時，在 WHERE 子句的子查詢中直接 SELECT FROM Person 做聚合或過濾。

DELETE from Person
where id NOT IN (
    select ID from (
        select min(id) as ID from Person
        group by email
    ) t
)


-- 176. Second Highest Salary
-- SELECT (sub_query) AS 'secondHighestsalary' 
SELECT COALESCE(
    (SELECT DISTINCT salary 
     FROM (
         SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk
         FROM Employee
     ) t
     WHERE rnk = 2),
    NULL
) AS `SecondHighestSalary`


-- 185. Department Top Three Salaries
-- CTE (Common Table Expressions) 通用資料表運算式
with tmp as (
    select d.name as Department, e.name as Employee, Salary, 
    dense_rank() over (partition by d.name order by e.salary desc) as rnk
    from Employee as e
    inner join Department as d
    on e.departmentId = d.id
)
select Department, Employee, Salary from tmp
where rnk <= 3


-- 1517. Find Users With Valid E-Mails
SELECT user_id, name, mail
FROM Users
-- Note that we also escaped the `@` character, as it has a special meaning in some regex flavors
WHERE mail REGEXP '^[a-zA-Z][a-zA-Z0-9_.-]*\\@leetcode\\.com$'


-- 1907. Count Salary Categories
-- SUM is critical here, since we need to take care of 0
-- The combination of Union and CASE
select "Low Salary" as category,
    SUM(CASE WHEN income < 20000 THEN 1 ELSE 0 END) as accounts_count
from accounts
union
select "Average Salary" as category,
    SUM(CASE WHEN income >= 20000 and income <= 50000 THEN 1 ELSE 0 END) as accounts_count
from accounts
union
select "High Salary" as category,
    SUM(CASE WHEN income > 50000 THEN 1 ELSE 0 END) as accounts_count
from account


-- 1484. Group Sold Products By The Date
-- very special, not that useful
select sell_date, count(distinct product) as num_sold, 
group_concat(distinct product order by product SEPARATOR ',') as products
from Activities
group by sell_date
order by sell_dat


-- 1204. Last Person to Fit in the Bus
select person_name from (
    select person_name, SUM(weight) over (order by turn) as total_weight
    from Queue
) t
where total_weight <= 1000
order by total_weight desc
limit 1


-- 1164. Product Price at a Given Date
WITH
  cte_price AS (
    SELECT
      product_id,
      new_price,
      ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY change_date DESC) AS rn
    FROM
      Products
    WHERE
      change_date <= '2019-08-16'
  )

SELECT
  DISTINCT Products.product_id,
  COALESCE(price.new_price, 10) AS price
FROM
  Products
LEFT JOIN
  cte_price AS price
ON
  Products.product_id = price.product_id
  AND price.rn = 1



CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
DECLARE M INT; 
    SET M = N-1; 
  RETURN (
      SELECT DISTINCT salary
      FROM Employee
      ORDER BY salary DESC
      LIMIT M, 1  -- index (rowcount-1), start from M+1=N, choose 1
      -- 0-indexed
      -- LIMIT offset, row_count
      -- 3 → 偏移量（offset)
      -- 1 → 回傳 1 行

      -- LIMIT row_count OFFSET offset
      -- 3 → 回傳 3 行
      -- OFFSET 1 → 跳過前 1 行

  );
END 


-- 2990. Loan Types
SELECT user_id 
FROM Loans
WHERE loan_type IN ('Refinance', 'Mortgage')
GROUP BY user_id
HAVING COUNT(DISTINCT loan_type) = 2
ORDER BY user_id as


-- 2987. Find Expensive Cities
select city from Listings
group by city
having avg(price) >  (select avg(price) from Listings)
order by cit


-- 181. Employees Earning More Than Their Managers
select e1.name as 'Employee'
from Employee e1
inner join Employee e2
on e1.managerId = e2.Id
where e1.salary > e2.salary


-- 586. Customer Placing the Largest Number of Orders
select customer_number
from Orders
group by customer_number
having count(order_number) = (
    SELECT count(order_number)
	FROM orders
	GROUP BY customer_number
	ORDER BY count(order_number) DESC LIMIT 1
)