"
normal func
agg func
window func

agg func --> NULL WON'T be included, but count will (count(*))
primary: unique + non-null

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

索引的類別分為 B-tree 與 Hash 2 種，這 2 種有各自適合的情境，譬如某些不重複的欄位，就適合使用 Hash 作為索引，不過 Hash 索引無法進行範圍查詢和排序，因此要考慮清楚0
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