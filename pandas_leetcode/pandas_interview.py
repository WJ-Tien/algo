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


def valid_emails(users: pd.DataFrame) -> pd.DataFrame:
    users = users[users['mail'].str.match(r'^[A-Za-z][A-Za-z0-9_.-]*@leetcode\.com$', na=False)]
    # if na=False --> NaN is replaced by False
    # if na=True --> NaN is NaN
    return users


def find_patients(patients: pd.DataFrame) -> pd.DataFrame:
    # 1527. Patients With a Condition
    df = patients[(patients.conditions.str.startswith("DIAB1")) | \
        (patients.conditions.str.contains(" DIAB1"))]
    return df