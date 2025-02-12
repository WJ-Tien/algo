"
normal func
agg func
window func

agg func --> NULL WON'T be included, but count will
primary: unique + non-null

是的，b.bonus IS NULL 這個條件是 必要的，
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

view -> virutual table --> select only -> update when querying 
materialized view (cache like) -> with real data -> need to update manually
store procedure (SP): sort of like function in SQL, can reduce IO/CPU workload
# IN(read), OUT(return the output), INOUT (best, bidirection)

ACID
Atomic: all successful or all failed
Consistency: e.g., bank balance must >= 0
isolation: each transaction is independant from each other
durability: store the data forever 

INDEX: B+ index (data distriubtion)
Partition (same table, e.g. hash partition)
Sharding (tables in different physical storage)
Clustering: store similar data in the neighboring physical address

"
--197. Rising Temperature
-- select w1.id from weather w1, weather w2 --> return all combs n^2 <- self-join
-- By doing a self-join on the Weather table, we create a Cartesian product of the table with itself, creating pairs of days
-- Postgresql
select w1.id from weather w1, weather w2
where w1.temperature > w2.temperature
and w1.recordDate::date - w2.recordDate::date = 1
-- MYSQL
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
select employee_id from Employees
where salary < 30000
and manager_id NOT IN (select employee_id from Employees)
order by employee_id