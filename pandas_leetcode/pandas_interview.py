import pandas as pd # noqa

def find_products(products: pd.DataFrame) -> pd.DataFrame:
    # Keep in mind, the return type expected is a pandas DataFrame; 
    # if you use single brackets instead of double brackets, 
    # you get a pandas Series (and the test fails).
    return products[(products.low_fats == 'Y') & (products.recyclable == 'Y')][['product_id']]


def find_customers(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
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