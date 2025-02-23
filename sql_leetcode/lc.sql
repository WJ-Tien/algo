"
normal func
agg func
window func

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

INDEX: B+ index (data distriubtion)
Partition (same table, e.g. hash partition)
Sharding (tables in different physical storage)
Clustering: store similar data in the neighboring physical address (e.g., disk)

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
FROM StudentSubjects ss
LEFT JOIN Examinations e 
ON ss.student_id = e.student_id 
AND ss.subject_name = e.subject_name
GROUP BY ss.student_id, ss.student_name, ss.subject_name
ORDER BY ss.student_id, ss.subject_name;


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
select p.project_id, ROUND(SUM(e.experience_years) / count(*), 2) as average_years
from Project as p
left join Employee as e
on p.employee_id = e.employee_id
group by project_id


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
SELECT DISTINCT employee_id, department_id
FROM Employee
WHERE employee_id IN (
    SELECT employee_id
    FROM Employee
    GROUP BY employee_id
    HAVING COUNT(*) = 1
  )
  OR primary_flag = 'Y'
ORDER BY employee_id


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
-- to prevent unknown table issue, an addtional subquery is required (see 't' sub table)
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
with tmp as (
    select d.name as Department, e.name as Employee, Salary, 
    dense_rank() over (partition by d.name order by e.salary desc) as rnk
    from Employee as e
    inner join Department as d
    on e.departmentId = d.id
)

select Department, Employee, Salary from tmp
where rnk <= 3