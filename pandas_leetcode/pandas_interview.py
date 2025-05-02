import numpy as np
import pandas as pd # noqa

"""
MUST groupby(['col_a'])['col_b'] --> return aggregated col_a & aggregated col_b
without col_b --> agg all columns except col_a
.agg('min') 
.agg(num_sold=('product', 'nunique'))

count --> non-nulls
size --> all (include nulls, series: #rows, dataframe: row * col)
agg funcs default would skip 0
In groupby.size() --> return #rows
count() --> need addtional cols, while size does not need
bool [True, False].sum() = 1 + 0
bool [True, False].count() = 2 
bool [True, False].value_counts() = True:1, False: 1

a = pd.Series({"A": 1, "B": 2, "C": 3}) --> series (index: ['A', 'B', 'C']) 
a = pd.DataFrame({"A": [1], "B": [2], "C": [3]}) --> dataframe

pd series can act as a python built-in array
e.g. set(pd.Series)

time series
rng = pd.date_range("1/1/2012", periods=100, freq="s")
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
ts2 = ts.resample("10s").mean()

df = pd.DataFrame({'value': [10, 20, 30, 40, 50]})

df['rolling_mean'] = df['value'].rolling(window=3).mean()
right is the start point. When you are at 30, you actually calculate 10<-20<-30
only keep data size == window

   value  rolling_mean
0     10           NaN
1     20           NaN
2     30          20.0
3     40          30.0
4     50          40.0

df['expanding_mean'] = df['value'].expanding().mean()
# cumulative, keep previous data
   value  expanding_mean
0     10            10.0
1     20            15.0
2     30            20.0
3     40            25.0
4     50            30.0


df = pd.DataFrame({"Col1": [10, 20, 15, 30, 45],
                    "Col2": [13, 23, 18, 33, 48],
                    "Col3": [17, 27, 22, 37, 52]},
                   index=pd.date_range("2020-01-01", "2020-01-05"))

shift(N)
N > 0 --> forward 
N < 0 --> backward
            Col1  Col2  Col3
2020-01-01    10    13    17
2020-01-02    20    23    27
2020-01-03    15    18    22
2020-01-04    30    33    37
2020-01-05    45    48    52

df.shift(periods=3)
            Col1  Col2  Col3
2020-01-01   NaN   NaN   NaN
2020-01-02   NaN   NaN   NaN
2020-01-03   NaN   NaN   NaN
2020-01-04  10.0  13.0  17.0
2020-01-05  20.0  23.0  27.0

df.shift(periods=1, axis="columns")
            Col1  Col2  Col3
2020-01-01   NaN    10    13
2020-01-02   NaN    20    23
2020-01-03   NaN    15    18
2020-01-04   NaN    30    33
2020-01-05   NaN    45    48

df.shift(periods=3, fill_value=0)
            Col1  Col2  Col3
2020-01-01     0     0     0
2020-01-02     0     0     0
2020-01-03     0     0     0
2020-01-04    10    13    17
2020-01-05    20    23    27

df.shift(periods=3, freq="D")
            Col1  Col2  Col3
2020-01-04    10    13    17
2020-01-05    20    23    27
2020-01-06    15    18    22
2020-01-07    30    33    37
2020-01-08    45    48    52

"""

def createDataframe(student_data: list[list[int]]) -> pd.DataFrame:
    # we can create a dataframe from a list
    # [col1_val, col2_val, ...]
    return pd.DataFrame(student_data, columns=['student_id', 'age'])


def changeDatatype(students: pd.DataFrame) -> pd.DataFrame:
    # df['column_name'] = df['column_name'].astype(new_dtype)
    students = students.astype({'grade': int})
    return students

def fillMissingValues(products: pd.DataFrame) -> pd.DataFrame:
    products['quantity'].fillna(0, inplace=True)
    # When we have multiple columns we can use dictionary,
    # products.fillna(value={'quantity':0},inplace=True)
    return products

def pivotTable(weather: pd.DataFrame) -> pd.DataFrame:
    # pivot: no duplicated items allowed, can't agg
    # pivot_table: long-format table to a wide-format table
    df = weather.pivot_table(index='month', columns='city', values='temperature', aggfunc='mean')
    return df

def find_products(products: pd.DataFrame) -> pd.DataFrame:
    # Keep in mind, the return type expected is a pandas DataFrame; 
    # if you use single brackets instead of double brackets, 
    # you get a pandas Series (and the test fails).
    return products[(products.low_fats == 'Y') & (products.recyclable == 'Y')][['product_id']]


def find_customers_never_order(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    # 183. Customers Who Never Order
    customers = customers.merge(orders, right_on=['customerId'], left_on=['id'], how='left')
    customers = customers[customers['customerId'].isnull()]
    customers.rename(columns={"name": "Customers"}, inplace=True)
    return customers[['Customers']]


def article_views(views: pd.DataFrame) -> pd.DataFrame:
    df = views[views.author_id == views.viewer_id]
    df.rename(columns={"author_id": "id"}, inplace=True)
    df = df.drop_duplicates(subset=['id']).sort_values(by=['id'])
    return df[['id']]


def invalid_tweets(tweets: pd.DataFrame) -> pd.DataFrame:
    df = tweets[tweets.content.str.len() > 15][['tweet_id']]
    return df


def calculate_special_bonus(employees: pd.DataFrame) -> pd.DataFrame:
    # 1873. Calculate Special Bonus

    def rule(row):
        if row['employee_id'] % 2 == 1 and not row['name'].startswith("M"):
            return row['salary']
        return 0
    
    employees['bonus'] = employees.apply(rule, axis=1)
    employees.sort_values(by=['employee_id'], inplace=True)

    return employees[['employee_id', 'bonus']]

    # employees['bonus'] = employees.apply(
    #     lambda row: row['salary'] if row['employee_id'] % 2 == 1 and not row['name'].startswith("M") else 0, 
    #     axis=1
    # )
    # employees.sort_values(by=['employee_id'], inplace=True)
    # return employees[['employee_id', 'bonus']


def fix_names(users: pd.DataFrame) -> pd.DataFrame:
    # 1667. Fix Names in a Table

    def fix_name(row):
        fixed_name = row['name'][0].upper() + row['name'][1:].lower()
        return fixed_name 
    
    users['name'] = users.apply(fix_name, axis=1)
    users.sort_values(by=['user_id'], inplace=True)
    return users[['user_id', 'name']]

def find_patients(patients: pd.DataFrame) -> pd.DataFrame:
    # 1527. Patients With a Condition
    df = patients[(patients.conditions.str.startswith("DIAB1")) | \
        (patients.conditions.str.contains(" DIAB1"))]
    return df

def valid_emails(users: pd.DataFrame) -> pd.DataFrame:
    users = users[users['mail'].str.match(r'^[A-Za-z][A-Za-z0-9_.-]*@leetcode\.com$', na=False)]
    # if na=False --> NaN is replaced by False
    # if na=True --> NaN is NaN
    # match --> only the prefix
    # fullmatch --> the entire string
    return users

def count_occurrences(files: pd.DataFrame) -> pd.DataFrame:
    # 2738. Count Occurrences in Text
    # case -> case-sensitive
    bear_count = files[
        files["content"].str.contains(r"(\s+bear\s+)", regex=True, case=False)
    ]["file_name"].nunique()

    bull_count = files[
        files["content"].str.contains(r"(\s+bull\s+)", regex=True, case=False)
    ]["file_name"].nunique()
    
    # Create result DataFrame
    result = pd.DataFrame({
        'word': ['bull', 'bear'],
        'count': [bull_count, bear_count]
    })
    
    return result



def nth_highest_salary(employee: pd.DataFrame, N: int) -> pd.DataFrame:
    # 177. Nth Highest Salary
    # 當 DataFrame 已經有行時：如果 DataFrame 已經有預先定義的行（index），那麼 df[col_name] = 1 會將這個值廣播（broadcast）到所有現有的行。
    # 當 DataFrame 是全新的空 DataFrame 時：如果 DataFrame 是剛創建的空 DataFrame，它沒有預先定義的行，這時 df[col_name] = 1 會創建列但不會創建任何行。
    column_name = f'getNthHighestSalary({N})'
    
    # 處理邊界情況
    if employee.empty or N <= 0:
        return pd.DataFrame({column_name: [None]})
    
    # 獲取唯一薪水值
    unique_salaries = employee['salary'].drop_duplicates()
    # unique_salaries: list = pd.Series(employee.salary.unique())

    
    # 檢查 N 是否大於唯一薪水的數量
    if N > len(unique_salaries):
        return pd.DataFrame({column_name: [None]})
    
    # 使用 nlargest 獲取第 N 高的薪水
    nth_salary = unique_salaries.nlargest(N).iloc[-1]
    
    # 創建一個有一行的 DataFrame
    return pd.DataFrame({column_name: [nth_salary]})

# def nth_highest_salary(employee: pd.DataFrame, N: int) -> pd.DataFrame:

#     unique_salaries: list = employee.salary.drop_duplicates()
#     col_name = f'getNthHighestSalary({N})'

#     if N <= 0 or len(unique_salaries) < N:
#         return pd.DataFrame({col_name: [None]})
#     else:
#         nth_salary = unique_salaries.nlargest(N).iloc[-1]
#         return pd.DataFrame({col_name: [nth_salary]})

def second_highest_salary(employee: pd.DataFrame) -> pd.DataFrame:
    unique_salaries = employee.salary.drop_duplicates().sort_values(ascending=False)

    if len(unique_salaries) < 2:
        return pd.DataFrame({"SecondHighestSalary": [None]})
    else:
        second_highest_salary = unique_salaries.iloc[1]
        return pd.DataFrame({"SecondHighestSalary": [second_highest_salary]})


def department_highest_salary(employee: pd.DataFrame, department: pd.DataFrame) -> pd.DataFrame:
    # 184. Department Highest Salary
    df = employee.merge(department, left_on="departmentId", right_on='id')
    df_max = df[df['salary'] == df.groupby('name_y')['salary'].transform('max')]
    df_max.rename(columns={"name_y":"Department", "name_x": "Employee", "salary": "Salary"}, inplace=True)
    return df_max[['Department', 'Employee', 'Salary']]

    # method == 'first' --> row_number 1 2 3
    # method == 'min' --> rank 11 3
    # method == 'dense' --> dense_rank 11 22 3


def order_scores(scores: pd.DataFrame) -> pd.DataFrame:
    # 178. Rank Scores
    scores['rank'] = scores['score'].rank(method='dense', ascending=False).astype(int)
    scores.sort_values(by=['rank'], inplace=True)
    return scores[['score', 'rank']]

    # partition over  
    # df['rank'] = df.groupby('group')['score'].transform('rank', method='dense', ascending=False)


def delete_duplicate_emails(person: pd.DataFrame) -> None:
    # 196. Delete Duplicate Emails
    person.sort_values(by=['id'], inplace=True)
    person.drop_duplicates(subset=['email'], keep='first', inplace=True)


def rearrange_products_table(products: pd.DataFrame) -> pd.DataFrame: 
    # 1795. Rearrange Products Table
    df = pd.melt(products, id_vars=['product_id'],
        value_vars=['store1', 'store2', 'store3'],
        var_name='store',
        value_name='price'
    ).dropna()
    return df

def count_rich_customers(store: pd.DataFrame) -> pd.DataFrame:
    # 2082. The Number of Rich Customers
    cnt = store[(store.amount > 500)]['customer_id'].drop_duplicates().count()
    # cnt = store[(store.amount > 500)]['customer_id'].nunique()
    return pd.DataFrame({"rich_count": [cnt]})


def count_salary_categories(accounts: pd.DataFrame) -> pd.DataFrame:

    conditions = [ accounts['income'] < 20000, 
                  (accounts['income'] >= 20000) & (accounts['income'] <= 50000),
                   accounts['income'] > 50000
    ]

    values = ["Low Salary", "Average Salary", "High Salary"]
    accounts['category'] = np.select(conditions, values, default="Unknown")
    df = accounts['category'].value_counts().reindex(values, fill_value=0).reset_index()
    df.columns = ['category', 'accounts_count']
    # accounts['category'].value_counts() --> pd.Series
    # reset_index --> transform to a normal dataframe
    # (accounts['income'] < 20000).count() --> 4, total len
    # .count() 計算的是「非空 (non-null) 值的個數」，而不是 True 的個數！
    # shape[0] calculate all
    # (accounts['income'] < 20000).sum() --> 1, (True==1, False == 0) 
    return df 


def ads_performance(ads: pd.DataFrame) -> pd.DataFrame: 
    # 1322. Ads Performance
    # very hard, although it's easy
    # apply to the normal dataframe --> elementwise
    # apply to the group dataframe from groupby --> groupwise
    ads['view_count'] = ads.groupby(['ad_id'])['action'].transform(lambda x: (x == 'Clicked').sum())
    ads['total_count'] = ads.groupby(['ad_id'])['action'].transform(lambda x: (x != 'Ignored').sum())
    ads['ctr'] = round(100 * ads['view_count'] / ads['total_count'], 2).fillna(0.0)
    ads.sort_values(by=['ctr', 'ad_id'], ascending=[False, True], inplace=True)
    ads.drop_duplicates(subset=['ad_id'], inplace=True)
    return ads[['ad_id', 'ctr']]

def ads_performance_2(ads: pd.DataFrame) -> pd.DataFrame: 
    # apply to the normal dataframe --> elementwise
    # apply to the group dataframe from groupby --> groupwise
    # by group process
    def calculate_ratio(g):
        return (100 * g['action'] == 'Clicked').sum() / (g['action'] != 'Ignored').sum()
    df = ads.groupby('ad_id').apply(calculate_ratio).reset_index()
    df.columns = ['ad_id', 'ctr']
    df['ctr'] = round(df['ctr'].fillna(0), 2)
    df.sort_values(['ctr', 'ad_id'], ascending=[False, True], inplace=True)
    return df

def total_time(employees: pd.DataFrame) -> pd.DataFrame:
    # 1741. Find Total Time Spent by Each Employee
    df = employees.groupby(['event_day', 'emp_id']).apply(lambda x: sum(x['out_time'] - x['in_time'])).reset_index()
    df.columns = ['day', 'emp_id', 'total_time']
    return df

def game_analysis(activity: pd.DataFrame) -> pd.DataFrame:
    df = activity.groupby(['player_id'])['event_date'].min().reset_index()
    df.rename(columns={"event_date": "first_login"}, inplace=True) 
    return df
    # df = activity.sort_values(by=['event_date']).drop_duplicates(subset=['player_id'], keep='first')
    # df.rename(columns={"event_date": "first_login"}, inplace=True) 
    # return df[['player_id', 'first_login']]

def largest_orders(orders: pd.DataFrame) -> pd.DataFrame:
    # 586. Customer Placing the Largest Number of Orders 
    df = orders.groupby(['customer_number'])['order_number'].size().reset_index()
    return df[df.order_number == df.order_number.max()][['customer_number']]

    def largest_orders(orders: pd.DataFrame) -> pd.DataFrame:
        df = orders.groupby(['customer_number']).size().reset_index(name='cnt')
        if df.empty:
            return pd.DataFrame({"customer_number": []}) # None (Null) != []
        largest = max(df['cnt'])
        return df[df.cnt == largest][['customer_number']]
   

def categorize_products(activities: pd.DataFrame) -> pd.DataFrame:
    # 1484. Group Sold Products By The Date
    def str_concat(g):
        g = sorted(g.drop_duplicates())
        return ','.join(g)
        
    df = activities.groupby(['sell_date']).agg(
        num_sold=('product', 'nunique'),
        products=('product', str_concat)
    ).reset_index()
    return df


def daily_leads_and_partners(daily_sales: pd.DataFrame) -> pd.DataFrame:
    # 1693. Daily Leads and Partners
    df = daily_sales.groupby(['date_id', 'make_name']).agg(
        unique_leads=("lead_id", 'nunique'), 
        unique_partners=("partner_id", 'nunique'), 
    ).reset_index()
    return df


def actors_and_directors(actor_director: pd.DataFrame) -> pd.DataFrame:
    # 1050. Actors and Directors Who Cooperated At Least Three Times
    # reset_index(name='?') will rename the groupbyed column
    df = actor_director.groupby(['actor_id', 'director_id']).size().reset_index(name='cnt')
    return df[df.cnt >= 3][['actor_id', 'director_id']]


def students_and_examinations(students: pd.DataFrame, subjects: pd.DataFrame, examinations: pd.DataFrame) -> pd.DataFrame:
    # 1280. Students and Examinations
    # cross join
    df = students.merge(subjects, how='cross')
    df2 = examinations.groupby(['student_id', 'subject_name']).size().reset_index(name='attended_exams')
    df = df.merge(df2, on=['student_id', 'subject_name'], how='left')
    df['attended_exams'].fillna(0, inplace=True)
    df.sort_values(by=['student_id', 'subject_name'], inplace=True)
    return df


def sales_person(sales_person: pd.DataFrame, company: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    # 607. Sales Person
    df = orders.merge(company, on=['com_id'], how='left')
    df.rename(columns={"name": "company_name"}, inplace=True)
    df = df.merge(sales_person, on=['sales_id'], how='right')
    red = df[df.company_name == 'RED']['name'].unique()
    return df[~df.name.isin(red)][['name']].drop_duplicates()

def loan_types(loans: pd.DataFrame) -> pd.DataFrame:
    df = loans[loans.loan_type.isin(['Refinance', 'Mortgage'])]
    df = df.groupby(['user_id'])['loan_type'].nunique().reset_index(name='unique_count')
    return df[df.unique_count >= 2][['user_id']]

def find_expensive_cities(listings: pd.DataFrame) -> pd.DataFrame:
    national_avg = listings['price'].mean()
    df = listings.groupby(['city'])['price'].mean().reset_index(name='city_avg')
    return df[df.city_avg > national_avg][['city']].sort_values(by=['city'], ascending=True)


def get_average_time(activity: pd.DataFrame) -> pd.DataFrame:
    # 1661. Average Time of Process per Machine
    df = activity.groupby(['machine_id']).apply(lambda x: (sum(x[x['activity_type']=='end']['timestamp']) - sum(x[x['activity_type']=='start']['timestamp'])) / x['process_id'].nunique()).reset_index(name='processing_time')
    df['processing_time'] = round(df['processing_time'], 3)
    return df[['machine_id', 'processing_time']]


def average_selling_price(prices: pd.DataFrame, units_sold: pd.DataFrame) -> pd.DataFrame:
    # 1251. Average Selling Price
    unique_product_id = prices.product_id.unique()

    # prices = pd.DataFrame(columns=['product_id'], data=prices['product_id'].unique())

    df = pd.merge(prices, units_sold, on=['product_id'], how='left')
    df = df[(df['start_date'] <= df['purchase_date']) & (df['purchase_date'] <= df['end_date'])]
    if df.shape[0] == 0:
        return pd.DataFrame({"product_id": unique_product_id, "average_price": [0]*len(unique_product_id)})

    df = df.groupby(['product_id']).apply(lambda x: sum(x['price'] * x['units'] / x['units'].sum()) if x['units'].sum() > 0 else 0).round(2).reset_index(name='average_price')
    df = prices[['product_id']].drop_duplicates().merge(df, on=['product_id'], how='left').fillna(0)
    return df


def rising_temperature(weather: pd.DataFrame) -> pd.DataFrame:
    # 197. Rising Temperature
    weather['recordDate'] = pd.to_datetime(weather['recordDate'])
    weather.sort_values('recordDate', inplace=True)
    weather['previousTemperature'] = weather['temperature'].shift(1)
    weather['previousDate'] = weather['recordDate'].shift(1)
    return weather[(weather.temperature > weather.previousTemperature) & 
                   (weather.recordDate == weather.previousDate + pd.Timedelta(days=1))][['id']]


def find_customers_no_trans(visits: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    # 1581. Customer Who Visited but Did Not Make Any Transactions
    df = pd.merge(visits, transactions, on=['visit_id'], how='left')
    df = df.groupby(['customer_id']).apply(lambda x: sum(x['transaction_id'].isnull())).reset_index(name='count_no_trans')
    return df[df.count_no_trans > 0]


def team_size(employee: pd.DataFrame) -> pd.DataFrame:
    # 1303. Find the Team Size
    # partition
    employee['team_size'] = employee.groupby(['team_id']).transform('size')
    return employee[['employee_id', 'team_size']]


def running_total(scores: pd.DataFrame) -> pd.DataFrame:
    # 1308. Running Total for Different Genders
    # over partition by order by
    # running total
    scores.sort_values(by=['gender', 'day'], ascending=True, inplace=True)
    scores['total'] = scores.groupby(['gender'])['score_points'].transform('cumsum')
    return scores[['gender', 'day', 'total']]


def find_employees(employees: pd.DataFrame) -> pd.DataFrame:
    # 1978. Employees Whose Manager Left the Company
    all_employees = employees.employee_id.unique()
    employees.sort_values(by=['employee_id'], ascending=True, inplace=True)
    return employees[(employees.salary < 30000) & (~employees.manager_id.isin(all_employees)) & (employees.manager_id.notnull())][['employee_id']]


def count_followers(followers: pd.DataFrame) -> pd.DataFrame:
    # 1729. Find Follwers Count
    return followers.groupby(['user_id']).size().reset_index(name='followers_count')


def project_employees_i(project: pd.DataFrame, employee: pd.DataFrame) -> pd.DataFrame:
    # 1075. Project Employees I
    df = pd.merge(project, employee, on=['employee_id'], how='left')
    df = df.groupby(['project_id'])['experience_years'].mean().round(2).reset_index(name='average_years')
    return df[['project_id', 'average_years']]


def find_employees_salary(employee: pd.DataFrame) -> pd.DataFrame:
    # 181. Employees Earning More Than Their Managers
    # inner join
    df = pd.merge(employee, employee, left_on=['managerId'], right_on=['id'])
    df.rename(columns={"name_x": 'Employee'}, inplace=True)
    return df[df.salary_x > df.salary_y][['Employee']]


def user_activity(activity: pd.DataFrame) -> pd.DataFrame:
    # 1141. User Activity for the Past 30 Days I 
    # between inclusive
    activity['activity_date'] = pd.to_datetime(activity['activity_date'])
    df = activity.groupby(['activity_date'])['user_id'].nunique().reset_index(name='active_users')
    df = df[df.activity_date.between((pd.Timestamp(2019,7,27) - pd.Timedelta(days=29)), pd.Timestamp(2019,7,27))]
    df.rename(columns={"activity_date": "day"}, inplace=True)
    return df

def users_percentage(users: pd.DataFrame, register: pd.DataFrame) -> pd.DataFrame:
    # 1633. Percentage of Users Attended a Contest
    total_cnt = users.user_id.nunique()
    df = (register.groupby(['contest_id'])['user_id'].size() * 100 / total_cnt).round(2).reset_index(name='percentage')
    df.sort_values(by=['percentage', 'contest_id'], ascending=[False, True], inplace=True)
    return df


def find_primary_department(employee: pd.DataFrame) -> pd.DataFrame:
    # 1789. Primary Department for Each Employee
    flag_n_eids = employee.groupby(['employee_id']).size().reset_index(name='cnt')
    flag_n_eids = flag_n_eids[flag_n_eids.cnt == 1]['employee_id'].unique()
    return employee[(employee.primary_flag == "Y") | (employee.employee_id.isin(flag_n_eids))][['employee_id', 'department_id']]