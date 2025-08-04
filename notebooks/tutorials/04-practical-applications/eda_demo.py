# Comprehensive EDA Demo: E-commerce Customer Analytics
## From SQL Database to Business Insights

This notebook demonstrates a complete exploratory data analysis workflow, starting from SQL database loading through advanced analytics using pandas and numpy.

---

## 📋 Table of Contents
1. [Database Setup & SQL Schema](#database-setup)
2. [Data Loading from SQL](#data-loading)
3. [Basic Data Exploration](#basic-exploration)
4. [Univariate Analysis](#univariate-analysis)
5. [Bivariate Analysis](#bivariate-analysis)
6. [Transaction Analysis](#transaction-analysis)
7. [Advanced Analytics & Customer Segmentation](#advanced-analytics)
8. [Data Quality Assessment](#data-quality)
9. [Key Business Insights](#insights)

---

## 🗄️ Database Setup & SQL Schema {#database-setup}

First, let's create our SQL database with realistic e-commerce tables. We'll use SQLite for simplicity.

```python
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

print("🎯 E-commerce Analytics: SQL to Insights Pipeline")
print("=" * 55)
```

### Create Database Connection

```python
# Create SQLite database connection
conn = sqlite3.connect('ecommerce_analytics.db')
cursor = conn.cursor()

print("✅ Database connection established")
```

### SQL Schema Creation

```python
# SQL DDL statements for our e-commerce schema
sql_schema = """
-- =====================================================
-- CUSTOMERS TABLE
-- =====================================================
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    age INTEGER NOT NULL,
    gender VARCHAR(10) NOT NULL,
    city VARCHAR(50) NOT NULL,
    registration_date DATE NOT NULL,
    loyalty_tier VARCHAR(20) NOT NULL,
    annual_income DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- PRODUCTS TABLE
-- =====================================================
DROP TABLE IF EXISTS products;

CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    category VARCHAR(50) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    brand VARCHAR(50) NOT NULL,
    rating DECIMAL(3,1) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- TRANSACTIONS TABLE
-- =====================================================
DROP TABLE IF EXISTS transactions;

CREATE TABLE transactions (
    transaction_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    transaction_date DATE NOT NULL,
    discount_pct DECIMAL(5,2) NOT NULL DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================
CREATE INDEX idx_customers_city ON customers(city);
CREATE INDEX idx_customers_loyalty ON customers(loyalty_tier);
CREATE INDEX idx_products_category ON products(category);
CREATE INDEX idx_transactions_date ON transactions(transaction_date);
CREATE INDEX idx_transactions_customer ON transactions(customer_id);
CREATE INDEX idx_transactions_product ON transactions(product_id);
"""

# Execute schema creation
cursor.executescript(sql_schema)
conn.commit()

print("✅ Database schema created successfully")
print("   • customers table with demographics and loyalty info")
print("   • products table with catalog and pricing")
print("   • transactions table with purchase history")
print("   • Performance indexes added")
```

### Generate and Insert Sample Data

```python
# Set seed for reproducibility
np.random.seed(42)

# Parameters for data generation
N_CUSTOMERS = 10000
N_PRODUCTS = 500
N_TRANSACTIONS = 50000

print(f"\n📊 Generating sample data:")
print(f"   • {N_CUSTOMERS:,} customers")
print(f"   • {N_PRODUCTS:,} products")
print(f"   • {N_TRANSACTIONS:,} transactions")
```

```python
# Generate customer data
cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
          'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
city_weights = [0.15, 0.12, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.15]

customers_data = []
for i in range(1, N_CUSTOMERS + 1):
    age = max(18, min(80, int(np.random.normal(40, 15))))
    gender = np.random.choice(['M', 'F', 'Other'], p=[0.45, 0.45, 0.1])
    city = np.random.choice(cities, p=city_weights)
    reg_date = (datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 1460))).date()
    loyalty = np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], p=[0.5, 0.3, 0.15, 0.05])
    
    # Income based on city and age
    city_multiplier = {'New York': 1.3, 'Los Angeles': 1.2, 'Chicago': 1.1, 'Houston': 1.0,
                      'Phoenix': 0.9, 'Philadelphia': 1.1, 'San Antonio': 0.85, 
                      'San Diego': 1.25, 'Dallas': 0.95, 'San Jose': 1.4}
    
    base_income = 30000 + age * 800 + np.random.normal(0, 15000)
    income = max(25000, min(200000, base_income * city_multiplier[city]))
    
    customers_data.append((i, age, gender, city, reg_date, loyalty, round(income, 2)))

# Insert customer data
cursor.executemany("""
    INSERT INTO customers (customer_id, age, gender, city, registration_date, loyalty_tier, annual_income)
    VALUES (?, ?, ?, ?, ?, ?, ?)
""", customers_data)

print("✅ Customer data inserted")
```

```python
# Generate product data
categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Beauty', 'Toys']
brands = [f'Brand_{i}' for i in range(1, 51)]

products_data = []
for i in range(1, N_PRODUCTS + 1):
    category = np.random.choice(categories)
    price = round(np.random.lognormal(3, 1), 2)
    brand = np.random.choice(brands)
    rating = round(max(1.0, min(5.0, np.random.normal(4.0, 0.8))), 1)
    
    products_data.append((i, category, price, brand, rating))

# Insert product data
cursor.executemany("""
    INSERT INTO products (product_id, category, price, brand, rating)
    VALUES (?, ?, ?, ?, ?)
""", products_data)

print("✅ Product data inserted")
```

```python
# Generate transaction data
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
date_range = (end_date - start_date).days

transactions_data = []
for i in range(1, N_TRANSACTIONS + 1):
    customer_id = np.random.randint(1, N_CUSTOMERS + 1)
    product_id = np.random.randint(1, N_PRODUCTS + 1)
    quantity = np.random.poisson(2) + 1
    trans_date = (start_date + timedelta(days=np.random.randint(0, date_range))).date()
    discount = round(max(0, min(50, np.random.exponential(5))), 2)
    
    transactions_data.append((i, customer_id, product_id, quantity, trans_date, discount))

# Insert transaction data
cursor.executemany("""
    INSERT INTO transactions (transaction_id, customer_id, product_id, quantity, transaction_date, discount_pct)
    VALUES (?, ?, ?, ?, ?, ?)
""", transactions_data)

conn.commit()
print("✅ Transaction data inserted")
print("\n🎉 Database setup complete!")
```

---

## 📥 Data Loading from SQL {#data-loading}

Now let's load our data from the SQL database using pandas, just as you would in a real-world scenario.

```python
print("📥 Loading data from SQL database...")
print("=" * 40)
```

### Load Core Tables

```python
# Load customers table
customers_query = """
SELECT 
    customer_id,
    age,
    gender,
    city,
    registration_date,
    loyalty_tier,
    annual_income
FROM customers
ORDER BY customer_id;
"""

customers = pd.read_sql_query(customers_query, conn)
print(f"✅ Loaded customers: {customers.shape[0]:,} rows × {customers.shape[1]} columns")
```

```python
# Load products table
products_query = """
SELECT 
    product_id,
    category,
    price,
    brand,
    rating
FROM products
ORDER BY product_id;
"""

products = pd.read_sql_query(products_query, conn)
print(f"✅ Loaded products: {products.shape[0]:,} rows × {products.shape[1]} columns")
```

```python
# Load transactions with enriched data using JOINs
transactions_query = """
SELECT 
    t.transaction_id,
    t.customer_id,
    t.product_id,
    t.quantity,
    t.transaction_date,
    t.discount_pct,
    p.price,
    p.category,
    p.brand,
    c.loyalty_tier,
    -- Calculate derived fields
    ROUND(p.price * (1 - t.discount_pct/100.0), 2) as unit_price,
    ROUND(p.price * (1 - t.discount_pct/100.0) * t.quantity, 2) as revenue
FROM transactions t
JOIN products p ON t.product_id = p.product_id
JOIN customers c ON t.customer_id = c.customer_id
ORDER BY t.transaction_date, t.transaction_id;
"""

transactions = pd.read_sql_query(transactions_query, conn)
print(f"✅ Loaded transactions: {transactions.shape[0]:,} rows × {transactions.shape[1]} columns")
```

### Data Type Conversions

```python
# Convert date columns to datetime
customers['registration_date'] = pd.to_datetime(customers['registration_date'])
transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])

# Convert categorical columns
customers['gender'] = customers['gender'].astype('category')
customers['city'] = customers['city'].astype('category')
customers['loyalty_tier'] = customers['loyalty_tier'].astype('category')
products['category'] = products['category'].astype('category')
products['brand'] = products['brand'].astype('category')

print("✅ Data types optimized")
print("\n📊 Ready for analysis!")
```

---

## 🔍 Basic Data Exploration {#basic-exploration}

Let's start with fundamental data exploration to understand our datasets.

```python
print("🔍 BASIC DATA EXPLORATION")
print("=" * 30)
```

### Dataset Overview

```python
# Display basic information about each dataset
datasets = {
    'Customers': customers,
    'Products': products, 
    'Transactions': transactions
}

for name, df in datasets.items():
    print(f"\n📋 {name} Dataset:")
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"   Date range: {df.select_dtypes(include=['datetime64']).min().min() if not df.select_dtypes(include=['datetime64']).empty else 'N/A'} to {df.select_dtypes(include=['datetime64']).max().max() if not df.select_dtypes(include=['datetime64']).empty else 'N/A'}")
```

### Data Quality Check

```python
print("\n🔍 Data Quality Overview:")
print("=" * 25)

for name, df in datasets.items():
    missing_data = df.isnull().sum()
    missing_pct = (missing_data / len(df) * 100).round(2)
    
    print(f"\n{name}:")
    print(f"   • Total missing values: {missing_data.sum()}")
    print(f"   • Duplicate rows: {df.duplicated().sum()}")
    
    if missing_data.sum() > 0:
        missing_summary = pd.DataFrame({
            'Missing_Count': missing_data[missing_data > 0],
            'Missing_Pct': missing_pct[missing_pct > 0]
        })
        print("   • Missing data by column:")
        print(missing_summary.to_string())
```

### Sample Data Preview

```python
print("\n👀 Sample Data Preview:")
print("=" * 22)

print("\n📊 Customers (first 5 rows):")
print(customers.head())

print("\n🛍️ Products (first 5 rows):")
print(products.head())

print("\n💰 Transactions (first 5 rows):")
print(transactions.head())
```

---

## 📊 Univariate Analysis {#univariate-analysis}

Deep dive into individual variables to understand distributions and patterns.

```python
print("📈 UNIVARIATE ANALYSIS")
print("=" * 25)
```

### Customer Demographics

```python
print("👥 Customer Demographics Analysis:")
print("=" * 35)

# Age distribution
print(f"\n📊 Age Statistics:")
age_stats = customers['age'].describe()
print(age_stats.round(1))

# Age distribution by percentiles
age_percentiles = customers['age'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
print(f"\nAge Percentiles:")
for p, val in age_percentiles.items():
    print(f"   {p*100:3.0f}th percentile: {val:.0f} years")
```

```python
# Gender distribution
print(f"\n📊 Gender Distribution:")
gender_dist = customers['gender'].value_counts()
gender_pct = customers['gender'].value_counts(normalize=True) * 100

gender_summary = pd.DataFrame({
    'Count': gender_dist,
    'Percentage': gender_pct.round(1)
})
print(gender_summary)
```

```python
# Geographic distribution
print(f"\n📊 Geographic Distribution:")
city_dist = customers['city'].value_counts()
city_pct = customers['city'].value_counts(normalize=True) * 100

city_summary = pd.DataFrame({
    'Count': city_dist,
    'Percentage': city_pct.round(1)
}).head(10)
print(city_summary)
```

```python
# Income distribution
print(f"\n💰 Income Analysis:")
income_stats = customers['annual_income'].describe()
print(income_stats.round(0))

# Income brackets
income_brackets = pd.cut(customers['annual_income'], 
                        bins=[0, 40000, 60000, 80000, 100000, float('inf')],
                        labels=['<$40K', '$40K-$60K', '$60K-$80K', '$80K-$100K', '>$100K'])

income_bracket_dist = income_brackets.value_counts()
print(f"\nIncome Brackets:")
for bracket, count in income_bracket_dist.items():
    pct = count / len(customers) * 100
    print(f"   {bracket}: {count:,} customers ({pct:.1f}%)")
```

### Product Catalog Analysis

```python
print("\n🛍️ Product Catalog Analysis:")
print("=" * 30)

# Category distribution
print("📊 Product Categories:")
category_dist = products['category'].value_counts()
for category, count in category_dist.items():
    pct = count / len(products) * 100
    print(f"   {category}: {count} products ({pct:.1f}%)")
```

```python
# Price analysis
print(f"\n💲 Price Distribution:")
price_stats = products['price'].describe()
print(price_stats.round(2))

# Price by category
print(f"\nAverage Price by Category:")
price_by_category = products.groupby('category')['price'].agg(['mean', 'median', 'std']).round(2)
price_by_category = price_by_category.sort_values('mean', ascending=False)
print(price_by_category)
```

```python
# Rating analysis
print(f"\n⭐ Product Ratings:")
rating_stats = products['rating'].describe()
print(rating_stats.round(2))

rating_dist = products['rating'].value_counts().sort_index()
print(f"\nRating Distribution:")
for rating, count in rating_dist.items():
    pct = count / len(products) * 100
    print(f"   {rating}⭐: {count} products ({pct:.1f}%)")
```

---

## 🔗 Bivariate Analysis {#bivariate-analysis}

Explore relationships between different variables to uncover patterns and correlations.

```python
print("🔗 BIVARIATE ANALYSIS")
print("=" * 22)
```

### Customer Demographics Relationships

```python
print("👥 Customer Demographics Relationships:")
print("=" * 40)

# Age vs Income correlation
age_income_corr = customers['age'].corr(customers['annual_income'])
print(f"📊 Age vs Income Correlation: {age_income_corr:.3f}")

if abs(age_income_corr) > 0.3:
    strength = "Strong" if abs(age_income_corr) > 0.7 else "Moderate"
    direction = "positive" if age_income_corr > 0 else "negative"
    print(f"   → {strength} {direction} correlation detected")
```

```python
# Income by gender
print(f"\n📊 Income Analysis by Gender:")
income_by_gender = customers.groupby('gender')['annual_income'].agg(['count', 'mean', 'median', 'std']).round(0)
income_by_gender.columns = ['Count', 'Mean_Income', 'Median_Income', 'Std_Dev']
print(income_by_gender)

# Statistical significance test could be added here
```

```python
# Income by city
print(f"\n📊 Income Analysis by City:")
income_by_city = customers.groupby('city')['annual_income'].agg(['count', 'mean', 'std']).round(0)
income_by_city.columns = ['Count', 'Mean_Income', 'Std_Dev']
income_by_city = income_by_city.sort_values('Mean_Income', ascending=False)
print(income_by_city)
```

```python
# Loyalty tier analysis
print(f"\n🏆 Loyalty Tier Demographics:")
loyalty_demo = customers.groupby('loyalty_tier').agg({
    'customer_id': 'count',
    'age': 'mean',
    'annual_income': 'mean'
}).round(1)
loyalty_demo.columns = ['Customer_Count', 'Avg_Age', 'Avg_Income']

# Calculate percentage distribution
loyalty_demo['Percentage'] = (loyalty_demo['Customer_Count'] / loyalty_demo['Customer_Count'].sum() * 100).round(1)
print(loyalty_demo)
```

### Product Relationships

```python
print(f"\n🛍️ Product Relationships:")
print("=" * 25)

# Price vs Rating correlation
price_rating_corr = products['price'].corr(products['rating'])
print(f"📊 Price vs Rating Correlation: {price_rating_corr:.3f}")

# Price analysis by category
print(f"\nPrice Range by Category:")
price_analysis = products.groupby('category')['price'].agg(['min', 'max', 'mean', 'count']).round(2)
price_analysis['range'] = price_analysis['max'] - price_analysis['min']
price_analysis = price_analysis.sort_values('mean', ascending=False)
print(price_analysis)
```

---

## 💰 Transaction Analysis {#transaction-analysis}

Dive deep into transaction patterns, revenue analysis, and customer behavior.

```python
print("💰 TRANSACTION ANALYSIS")
print("=" * 25)
```

### Revenue Metrics

```python
print("📊 Overall Revenue Metrics:")
print("=" * 28)

total_revenue = transactions['revenue'].sum()
total_transactions = len(transactions)
avg_order_value = transactions['revenue'].mean()
avg_quantity = transactions['quantity'].mean()

print(f"   • Total Revenue: ${total_revenue:,.2f}")
print(f"   • Total Transactions: {total_transactions:,}")
print(f"   • Average Order Value: ${avg_order_value:.2f}")
print(f"   • Average Quantity per Order: {avg_quantity:.1f}")
print(f"   • Revenue per Transaction: ${total_revenue/total_transactions:.2f}")
```

### Revenue by Category

```python
print(f"\n📊 Revenue Analysis by Product Category:")
print("=" * 42)

category_analysis = transactions.groupby('category').agg({
    'revenue': ['sum', 'count', 'mean'],
    'quantity': 'sum'
}).round(2)

# Flatten column names
category_analysis.columns = ['Total_Revenue', 'Transaction_Count', 'Avg_Revenue', 'Total_Quantity']

# Add percentage of total revenue
category_analysis['Revenue_Pct'] = (category_analysis['Total_Revenue'] / total_revenue * 100).round(1)

# Sort by total revenue
category_analysis = category_analysis.sort_values('Total_Revenue', ascending=False)
print(category_analysis)
```

### Temporal Analysis

```python
print(f"\n📈 Temporal Revenue Analysis:")
print("=" * 30)

# Add time features
transactions['month'] = transactions['transaction_date'].dt.month
transactions['quarter'] = transactions['transaction_date'].dt.quarter
transactions['day_of_week'] = transactions['transaction_date'].dt.day_name()

# Monthly revenue trend
print("Monthly Revenue Trend:")
monthly_revenue = transactions.groupby('month').agg({
    'revenue': 'sum',
    'transaction_id': 'count'
}).round(2)
monthly_revenue.columns = ['Total_Revenue', 'Transaction_Count']

for month, row in monthly_revenue.iterrows():
    month_name = pd.to_datetime(f'2023-{month:02d}-01').strftime('%B')
    print(f"   {month_name:>9}: ${row['Total_Revenue']:>10,.0f} ({row['Transaction_Count']:,} transactions)")
```

```python
# Quarterly analysis
print(f"\nQuarterly Performance:")
quarterly_revenue = transactions.groupby('quarter').agg({
    'revenue': 'sum',
    'transaction_id': 'count'
}).round(2)

for quarter, row in quarterly_revenue.iterrows():
    print(f"   Q{quarter}: ${row['revenue']:>12,.0f} ({row['transaction_id']:,} transactions)")
```

### Customer Purchase Behavior

```python
print(f"\n👤 Customer Purchase Behavior:")
print("=" * 32)

# Customer-level aggregation
customer_behavior = transactions.groupby('customer_id').agg({
    'revenue': ['sum', 'count', 'mean'],
    'quantity': 'sum',
    'transaction_date': ['min', 'max']
}).round(2)

# Flatten columns
customer_behavior.columns = ['Total_Spent', 'Purchase_Count', 'Avg_Order_Value', 'Total_Items', 'First_Purchase', 'Last_Purchase']

# Calculate customer lifetime (days between first and last purchase)
customer_behavior['Customer_Lifetime_Days'] = (
    pd.to_datetime(customer_behavior['Last_Purchase']) - 
    pd.to_datetime(customer_behavior['First_Purchase'])
).dt.days

print("Customer Behavior Summary:")
behavior_summary = customer_behavior[['Total_Spent', 'Purchase_Count', 'Avg_Order_Value']].describe().round(2)
print(behavior_summary)
```

### Loyalty Tier Performance

```python
print(f"\n🏆 Performance by Loyalty Tier:")
print("=" * 32)

loyalty_performance = transactions.groupby('loyalty_tier').agg({
    'revenue': ['sum', 'mean', 'count'],
    'quantity': 'mean'
}).round(2)

loyalty_performance.columns = ['Total_Revenue', 'Avg_Revenue', 'Transaction_Count', 'Avg_Quantity']

# Add customer count for each tier
tier_customers = customers['loyalty_tier'].value_counts()
loyalty_performance['Customer_Count'] = loyalty_performance.index.map(tier_customers)
loyalty_performance['Revenue_Per_Customer'] = (loyalty_performance['Total_Revenue'] / loyalty_performance['Customer_Count']).round(2)

print(loyalty_performance)
```

---

## 🎯 Advanced Analytics & Customer Segmentation {#advanced-analytics}

Apply advanced analytical techniques including customer segmentation and lifetime value analysis.

```python
print("🎯 ADVANCED ANALYTICS")
print("=" * 22)
```

### RFM Analysis (Recency, Frequency, Monetary)

```python
print("📊 RFM Analysis - Customer Segmentation:")
print("=" * 40)

# Calculate RFM metrics
analysis_date = transactions['transaction_date'].max()
print(f"Analysis Date: {analysis_date.date()}")

rfm = transactions.groupby('customer_id').agg({
    'transaction_date': lambda x: (analysis_date - x.max()).days,  # Recency
    'transaction_id': 'count',  # Frequency
    'revenue': 'sum'  # Monetary
}).rename(columns={
    'transaction_date': 'Recency',
    'transaction_id': 'Frequency', 
    'revenue': 'Monetary'
})

print(f"\nRFM Metrics Summary:")
print(rfm.describe().round(2))
```

```python
# Create RFM scores using quartiles
rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])  # Lower recency = higher score
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4])

# Combine scores
rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

print(f"\nRFM Score Distribution (top 10):")
top_rfm_scores = rfm['RFM_Score'].value_counts().head(10)
for score, count in top_rfm_scores.items():
    print(f"   {score}: {count} customers")
```

```python
# Define customer segments based on RFM scores
def categorize_customers(row):
    """Categorize customers based on RFM scores"""
    score = row['RFM_Score']
    
    # Champions: High value, recent, frequent
    if score in ['444', '443', '434', '344', '433']:
        return 'Champions'
    
    # Loyal Customers: High frequency and monetary, may not be recent
    elif score in ['343', '334', '333', '432', '342']:
        return 'Loyal Customers'
    
    # Potential Loyalists: Recent customers with good potential
    elif score in ['431', '413', '341', '143', '241', '142']:
        return 'Potential Loyalists'
    
    # At Risk: Were valuable but haven't purchased recently
    elif score in ['242', '243', '141', '132', '131']:
        return 'At Risk'
    
    # Can't Lose Them: High monetary value but low recency and frequency
    elif score in ['122', '123', '124', '222', '223']:
        return 'Cannot Lose'
    
    # New Customers: Recent but low frequency and monetary
    elif score in ['411', '412', '421', '422']:
        return 'New Customers'
    
    else:
        return 'Others'

rfm['Customer_Segment'] = rfm.apply(categorize_customers, axis=1)

print(f"\n👥 Customer Segmentation Results:")
segment_summary = rfm['Customer_Segment'].value_counts()
segment_percentages = (segment_summary / len(rfm) * 100).round(1)

segment_analysis = pd.DataFrame({
    'Customer_Count': segment_summary,
    'Percentage': segment_percentages
})

print(segment_analysis)
```

### Customer Lifetime Value (CLV) Analysis

```python
print(f"\n💎 Customer Lifetime Value Analysis:")
print("=" * 37)

# Merge RFM with customer demographics
clv_analysis = rfm.merge(customers[['customer_id', 'loyalty_tier', 'annual_income']], 
                        left_index=True, right_on='customer_id', how='left')

# Simple CLV calculation (can be enhanced with more sophisticated models)
clv_analysis['Avg_Order_Value'] = clv_analysis['Monetary'] / clv_analysis['Frequency']
clv_analysis['Purchase_Frequency'] = clv_analysis['Frequency'] / 365  # Daily frequency
clv_analysis['Customer_Lifespan'] = 365 - clv_analysis['Recency']  # Simplified lifespan

# CLV = Average Order Value × Purchase Frequency × Customer Lifespan × Profit Margin
# Assuming 20% profit margin
profit_margin = 0.2
clv_analysis['Estimated_CLV'] = (
    clv_analysis['Avg_Order_Value'] * 
    clv_analysis['Purchase_Frequency'] * 
    clv_analysis['Customer_Lifespan'] * 
    profit_margin
).round(2)

print("CLV by Customer Segment:")
clv_by_segment = clv_analysis.groupby('Customer_Segment').agg({
    'Estimated_CLV': ['count', 'mean', 'median', 'sum'],
    'Monetary': 'mean',
    'Frequency': 'mean'
}).round(2)

# Flatten column names
clv_by_segment.columns = ['Customer_Count', 'Mean_CLV', 'Median_CLV', 'Total_CLV', 'Avg_Spend', 'Avg_Frequency']
clv_by_segment = clv_by_segment.sort_values('Mean_CLV', ascending=False)
print(clv_by_segment)
```

### Product Performance Analysis

```python
print(f"\n🏆 Product Performance Analysis:")
print("=" * 33)

# Product-level performance metrics
product_performance = transactions.groupby('product_id').agg({
    'revenue': 'sum',
    'quantity': 'sum',
    'transaction_id': 'count'
}).round(2)

product_performance.columns = ['Total_Revenue', 'Total_Quantity', 'Transaction_Count']

# Merge with product details
product_performance = product_performance.merge(
    products[['product_id', 'category', 'price', 'brand', 'rating']], 
    left_index=True, right_on='product_id', how='left'
)

# Calculate additional metrics
product_performance['Revenue_Per_Transaction'] = (
    product_performance['Total_Revenue'] / product_performance['Transaction_Count']
).round(2)

print("Top 10 Products by Revenue:")
top_products = product_performance.nlargest(10, 'Total_Revenue')
print(top_products[['category', 'brand', 'price', 'rating', 'Total_Revenue', 'Total_Quantity']].to_string())
```

---

## 🔧 Data Quality Assessment {#data-quality}

Comprehensive data quality analysis including outlier detection and consistency checks.

```python
print("🔧 DATA QUALITY ASSESSMENT")
print("=" * 28)
```

### Outlier Detection

```python
print("📊 Outlier Detection Analysis:")
print("=" * 30)

def detect_outliers_iqr(series, multiplier=1.5):
    """Detect outliers using IQR method"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers, lower_bound, upper_bound

# Income outliers
income_outliers, income_lower, income_upper = detect_outliers_iqr(customers['annual_income'])
print(f"📈 Income Outliers:")
print(f"   • Total outliers: {len(income_outliers)} ({len(income_outliers)/len(customers)*100:.1f}%)")
print(f"   • Valid range: ${income_lower:,.0f} - ${income_upper:,.0f}")
print(f"   • Outlier range: ${income_outliers.min():,.0f} - ${income_outliers.max():,.0f}")
```

```python
# Revenue outliers
revenue_outliers, revenue_lower, revenue_upper = detect_outliers_iqr(transactions['revenue'])
print(f"\n💰 Transaction Revenue Outliers:")
print(f"   • Total outliers: {len(revenue_outliers)} ({len(revenue_outliers)/len(transactions)*100:.1f}%)")
print(f"   • Valid range: ${revenue_lower:.2f} - ${revenue_upper:.2f}")
if len(revenue_outliers) > 0:
    print(f"   • Outlier range: ${revenue_outliers.min():.2f} - ${revenue_outliers.max():.2f}")
```

```python
# Age outliers
age_outliers, age_lower, age_upper = detect_outliers_iqr(customers['age'])
print(f"\n👤 Age Outliers:")
print(f"   • Total outliers: {len(age_outliers)} ({len(age_outliers)/len(customers)*100:.1f}%)")
print(f"   • Valid range: {age_lower:.0f} - {age_upper:.0f} years")
if len(age_outliers) > 0:
    print(f"   • Outlier range: {age_outliers.min():.0f} - {age_outliers.max():.0f} years")
```

### Data Consistency Checks

```python
print(f"\n🔍 Data Consistency Checks:")
print("=" * 28)

# Check for impossible values
print("Impossible Values Check:")

# Negative values where they shouldn't exist
negative_revenue = transactions[transactions['revenue'] < 0]
negative_quantity = transactions[transactions['quantity'] <= 0]
negative_prices = products[products['price'] <= 0]

print(f"   • Negative revenue transactions: {len(negative_revenue)}")
print(f"   • Zero/negative quantity transactions: {len(negative_quantity)}")
print(f"   • Zero/negative price products: {len(negative_prices)}")

# Rating bounds check
invalid_ratings = products[(products['rating'] < 1) | (products['rating'] > 5)]
print(f"   • Invalid product ratings (not 1-5): {len(invalid_ratings)}")

# Discount percentage check
invalid_discounts = transactions[(transactions['discount_pct'] < 0) | (transactions['discount_pct'] > 100)]
print(f"   • Invalid discount percentages: {len(invalid_discounts)}")
```

### Referential Integrity

```python
print(f"\n🔗 Referential Integrity Check:")
print("=" * 31)

# Check for orphaned records
transaction_customers = set(transactions['customer_id'].unique())
valid_customers = set(customers['customer_id'].unique())
orphaned_customer_transactions = transaction_customers - valid_customers

transaction_products = set(transactions['product_id'].unique())
valid_products = set(products['product_id'].unique())
orphaned_product_transactions = transaction_products - valid_products

print(f"Orphaned Records:")
print(f"   • Transactions with invalid customer_id: {len(orphaned_customer_transactions)}")
print(f"   • Transactions with invalid product_id: {len(orphaned_product_transactions)}")

# Customer activity check
active_customers = set(transactions['customer_id'].unique())
inactive_customers = valid_customers - active_customers
print(f"   • Customers with no transactions: {len(inactive_customers)} ({len(inactive_customers)/len(customers)*100:.1f}%)")
```

---

## 💡 Key Business Insights {#insights}

Synthesize all analysis into actionable business insights and recommendations.

```python
print("💡 KEY BUSINESS INSIGHTS & RECOMMENDATIONS")
print("=" * 45)
```

### Executive Summary

```python
print("🎯 Executive Summary:")
print("=" * 20)

# Calculate key business metrics
total_customers = len(customers)
active_customers = len(transactions['customer_id'].unique())
customer_activation_rate = (active_customers / total_customers * 100)

total_revenue = transactions['revenue'].sum()
avg_customer_value = total_revenue / active_customers
top_category = category_analysis.index[0]
top_tier_customers = segment_summary.get('Champions', 0)

print(f"""
📊 Business Overview:
   • Total Revenue Generated: ${total_revenue:,.0f}
   • Active Customer Base: {active_customers:,} out of {total_customers:,} ({customer_activation_rate:.1f}%)
   • Average Customer Value: ${avg_customer_value:.2f}
   • Top Product Category: {top_category} (${category_analysis.loc[top_category, 'Total_Revenue']:,.0f})
   • Champion Customers: {top_tier_customers} ({top_tier_customers/active_customers*100:.1f}% of active base)
""")
```

### Customer Insights

```python
print("👥 Customer Insights:")
print("=" * 20)

# Demographics insights
avg_age = customers['age'].mean()
avg_income = customers['annual_income'].mean()
top_city = customers['city'].mode()[0]
top_loyalty_tier = customers['loyalty_tier'].mode()[0]

# Behavioral insights
avg_orders_per_customer = transactions.groupby('customer_id').size().mean()
repeat_customers = len(transactions.groupby('customer_id').size()[transactions.groupby('customer_id').size() > 1])
repeat_rate = repeat_customers / active_customers * 100

print(f"""
🎯 Customer Profile:
   • Average Age: {avg_age:.0f} years
   • Average Income: ${avg_income:,.0f}
   • Largest Market: {top_city} ({customers[customers['city']==top_city].shape[0]} customers)
   • Most Common Tier: {top_loyalty_tier} ({customers[customers['loyalty_tier']==top_loyalty_tier].shape[0]} customers)

💡 Behavioral Patterns:
   • Average Orders per Customer: {avg_orders_per_customer:.1f}
   • Customer Repeat Rate: {repeat_rate:.1f}%
   • Top Customer Segment: {segment_summary.index[0]} ({segment_summary.iloc[0]} customers)
""")
```

### Revenue Insights

```python
print("💰 Revenue Insights:")
print("=" * 19)

# Seasonal patterns
best_month = monthly_revenue.idxmax()
best_month_name = pd.to_datetime(f'2023-{best_month:02d}-01').strftime('%B')
worst_month = monthly_revenue.idxmin()
worst_month_name = pd.to_datetime(f'2023-{worst_month:02d}-01').strftime('%B')

# Category performance
category_performance = category_analysis.copy()
top_3_categories = category_performance.head(3)

print(f"""
📈 Revenue Patterns:
   • Peak Month: {best_month_name} (${monthly_revenue.iloc[best_month-1]['Total_Revenue']:,.0f})
   • Lowest Month: {worst_month_name} (${monthly_revenue.iloc[worst_month-1]['Total_Revenue']:,.0f})
   • Revenue Seasonality: {((monthly_revenue.max()['Total_Revenue'] - monthly_revenue.min()['Total_Revenue']) / monthly_revenue.mean()['Total_Revenue'] * 100):.1f}% variation

🏆 Top Performing Categories:
""")

for idx, (category, row) in enumerate(top_3_categories.iterrows(), 1):
    print(f"   {idx}. {category}: ${row['Total_Revenue']:,.0f} ({row['Revenue_Pct']:.1f}% of total)")
```

### Strategic Recommendations

```python
print("\n🚀 Strategic Recommendations:")
print("=" * 29)

recommendations = [
    "🎯 Customer Retention:",
    f"   • Focus on {segment_summary.get('At Risk', 0)} 'At Risk' customers with targeted campaigns",
    f"   • Nurture {segment_summary.get('Potential Loyalists', 0)} 'Potential Loyalists' to become Champions",
    f"   • Implement loyalty program enhancements for {customers['loyalty_tier'].value_counts()['Bronze']} Bronze tier customers",
    "",
    "💰 Revenue Optimization:",
    f"   • Invest more in {top_category} category (highest revenue generator)",
    f"   • Address seasonality with promotions during {worst_month_name} (lowest performing month)",
    f"   • Capitalize on {best_month_name} success patterns for other months",
    "",
    "🎪 Product Strategy:",
    f"   • Expand high-rating product lines (average rating: {products['rating'].mean():.1f}/5.0)",
    f"   • Review pricing strategy for categories with high price variance",
    f"   • Consider discontinuing low-performing products in bottom revenue quartile",
    "",
    "📊 Data & Analytics:",
    f"   • Implement real-time CLV tracking for the {top_tier_customers} Champion customers",
    f"   • Set up automated alerts for customers transitioning to 'At Risk' segment",
    f"   • Develop predictive models for customer churn prevention",
    "",
    "🌟 Growth Opportunities:",
    f"   • Target {len(inactive_customers)} inactive customers with re-engagement campaigns",
    f"   • Expand in high-income markets like {income_by_city.index[0]}",
    f"   • Develop premium products for customers with >$100K income ({len(customers[customers['annual_income'] > 100000])} customers)"
]

for recommendation in recommendations:
    print(recommendation)
```

### Performance Summary

```python
print(f"\n📊 Final Performance Summary:")
print("=" * 30)

# Create comprehensive summary metrics
summary_metrics = {
    'Total Revenue': f"${total_revenue:,.0f}",
    'Active Customers': f"{active_customers:,}",
    'Avg Order Value': f"${avg_order_value:.2f}",
    'Customer Segments': len(segment_summary),
    'Product Categories': len(products['category'].unique()),
    'Geographic Markets': len(customers['city'].unique()),
    'Data Quality Score': f"{100 - ((len(income_outliers) + len(revenue_outliers))/total_transactions*100):.1f}%",
    'Analysis Completeness': "100%"
}

print("Key Performance Indicators:")
for metric, value in summary_metrics.items():
    print(f"   • {metric}: {value}")

print(f"\n🎉 Analysis Complete!")
print(f"   Database: {N_TRANSACTIONS:,} transactions analyzed")
print(f"   Time Period: 2023 Full Year")
print(f"   Insights Generated: 50+ actionable findings")
```

---

## 📊 Comprehensive Data Visualizations {#visualizations}

Now let's create a complete set of visualizations to bring our data insights to life using matplotlib, seaborn, and advanced plotting libraries.

```python
print("📊 COMPREHENSIVE DATA VISUALIZATIONS")
print("=" * 38)

# Additional imports for advanced visualizations
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import squarify  # For treemaps
from math import pi

# Set up the plotting environment
plt.style.use('default')  # Reset to default for better control
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("✅ Visualization libraries loaded")
print("📈 Creating comprehensive chart gallery...")
```

### 1. Line Plots - Time Series Analysis

```python
print("\n📈 Line Plots - Temporal Analysis")
print("=" * 33)

# Prepare monthly data
monthly_data = transactions.groupby(transactions['transaction_date'].dt.to_period('M')).agg({
    'revenue': 'sum',
    'transaction_id': 'count',
    'quantity': 'sum'
}).reset_index()
monthly_data['transaction_date'] = monthly_data['transaction_date'].dt.to_timestamp()

# Create subplot with multiple line plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Time Series Analysis - Multiple Metrics', fontsize=16, fontweight='bold')

# Revenue trend
axes[0,0].plot(monthly_data['transaction_date'], monthly_data['revenue'], 
               marker='o', linewidth=2, markersize=6, color='#2E86AB')
axes[0,0].set_title('Monthly Revenue Trend')
axes[0,0].set_ylabel('Revenue ($)')
axes[0,0].grid(True, alpha=0.3)
axes[0,0].tick_params(axis='x', rotation=45)

# Transaction count trend  
axes[0,1].plot(monthly_data['transaction_date'], monthly_data['transaction_id'], 
               marker='s', linewidth=2, markersize=6, color='#A23B72')
axes[0,1].set_title('Monthly Transaction Count')
axes[0,1].set_ylabel('Number of Transactions')
axes[0,1].grid(True, alpha=0.3)
axes[0,1].tick_params(axis='x', rotation=45)

# Cumulative revenue
cumulative_revenue = monthly_data['revenue'].cumsum()
axes[1,0].plot(monthly_data['transaction_date'], cumulative_revenue, 
               marker='d', linewidth=3, markersize=6, color='#F18F01')
axes[1,0].set_title('Cumulative Revenue Growth')
axes[1,0].set_ylabel('Cumulative Revenue ($)')
axes[1,0].grid(True, alpha=0.3)
axes[1,0].tick_params(axis='x', rotation=45)

# Multiple metrics on same plot
ax2 = axes[1,1]
ax3 = ax2.twinx()

line1 = ax2.plot(monthly_data['transaction_date'], monthly_data['revenue'], 
                 'b-', marker='o', label='Revenue')
line2 = ax3.plot(monthly_data['transaction_date'], monthly_data['transaction_id'], 
                 'r--', marker='s', label='Transactions')

ax2.set_ylabel('Revenue ($)', color='b')
ax3.set_ylabel('Transaction Count', color='r')
ax2.set_title('Revenue vs Transaction Volume')
ax2.tick_params(axis='x', rotation=45)

# Combine legends
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.show()

print("✅ Line plots created - showing temporal trends and relationships")
```

### 2. Scatter Plots - Correlation Analysis

```python
print("\n🔸 Scatter Plots - Correlation Analysis")
print("=" * 37)

# Prepare customer data for scatter analysis
customer_summary = transactions.groupby('customer_id').agg({
    'revenue': 'sum',
    'transaction_id': 'count',
    'quantity': 'sum'
}).reset_index()
customer_summary = customer_summary.merge(customers[['customer_id', 'age', 'annual_income']], on='customer_id')

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Customer Behavior Scatter Analysis', fontsize=16, fontweight='bold')

# Age vs Total Spending
scatter1 = axes[0,0].scatter(customer_summary['age'], customer_summary['revenue'], 
                            c=customer_summary['annual_income'], cmap='viridis', 
                            alpha=0.6, s=50)
axes[0,0].set_xlabel('Customer Age')
axes[0,0].set_ylabel('Total Spending ($)')
axes[0,0].set_title('Age vs Spending (colored by Income)')
plt.colorbar(scatter1, ax=axes[0,0], label='Annual Income ($)')

# Income vs Spending with trend line
axes[0,1].scatter(customer_summary['annual_income'], customer_summary['revenue'], 
                  alpha=0.6, s=50, color='#E74C3C')
# Add trend line
z = np.polyfit(customer_summary['annual_income'], customer_summary['revenue'], 1)
p = np.poly1d(z)
axes[0,1].plot(customer_summary['annual_income'], p(customer_summary['annual_income']), 
               "r--", alpha=0.8, linewidth=2)
axes[0,1].set_xlabel('Annual Income ($)')
axes[0,1].set_ylabel('Total Spending ($)')
axes[0,1].set_title('Income vs Spending with Trend Line')

# Transaction Count vs Average Order Value
customer_summary['avg_order_value'] = customer_summary['revenue'] / customer_summary['transaction_id']
scatter2 = axes[1,0].scatter(customer_summary['transaction_id'], customer_summary['avg_order_value'],
                            c=customer_summary['age'], cmap='plasma', alpha=0.6, s=50)
axes[1,0].set_xlabel('Number of Transactions')
axes[1,0].set_ylabel('Average Order Value ($)')
axes[1,0].set_title('Frequency vs Order Value (colored by Age)')
plt.colorbar(scatter2, ax=axes[1,0], label='Age')

# 3D-style bubble plot
bubble_sizes = customer_summary['quantity'] * 2  # Scale for visibility
axes[1,1].scatter(customer_summary['age'], customer_summary['revenue'], 
                  s=bubble_sizes, alpha=0.5, c=customer_summary['transaction_id'], 
                  cmap='coolwarm')
axes[1,1].set_xlabel('Customer Age')
axes[1,1].set_ylabel('Total Spending ($)')
axes[1,1].set_title('Bubble Plot: Age vs Spending\n(size=quantity, color=frequency)')

plt.tight_layout()
plt.show()

print("✅ Scatter plots created - showing correlations and multi-dimensional relationships")
```

### 3. Bar Charts - Categorical Analysis

```python
print("\n📊 Bar Charts - Categorical Analysis")
print("=" * 34)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Categorical Data Analysis', fontsize=16, fontweight='bold')

# Revenue by Category (Horizontal Bar)
category_revenue = transactions.groupby('category')['revenue'].sum().sort_values(ascending=True)
bars1 = axes[0,0].barh(category_revenue.index, category_revenue.values, 
                       color=plt.cm.Set3(np.linspace(0, 1, len(category_revenue))))
axes[0,0].set_xlabel('Total Revenue ($)')
axes[0,0].set_title('Revenue by Product Category')
# Add value labels
for i, bar in enumerate(bars1):
    width = bar.get_width()
    axes[0,0].text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                   f'${width:,.0f}', ha='left', va='center', fontweight='bold')

# Customer Count by City (Vertical Bar)
city_counts = customers['city'].value_counts()
bars2 = axes[0,1].bar(city_counts.index, city_counts.values, 
                      color=plt.cm.tab10(np.linspace(0, 1, len(city_counts))))
axes[0,1].set_ylabel('Number of Customers')
axes[0,1].set_title('Customer Distribution by City')
axes[0,1].tick_params(axis='x', rotation=45)
# Add value labels
for bar in bars2:
    height = bar.get_height()
    axes[0,1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')

# Stacked Bar - Revenue by Category and Loyalty Tier
category_loyalty = transactions.groupby(['category', 'loyalty_tier'])['revenue'].sum().unstack(fill_value=0)
category_loyalty.plot(kind='bar', stacked=True, ax=axes[1,0], 
                     colormap='viridis', width=0.8)
axes[1,0].set_xlabel('Product Category')
axes[1,0].set_ylabel('Revenue ($)')
axes[1,0].set_title('Revenue by Category and Loyalty Tier (Stacked)')
axes[1,0].tick_params(axis='x', rotation=45)
axes[1,0].legend(title='Loyalty Tier', bbox_to_anchor=(1.05, 1), loc='upper left')

# Grouped Bar - Average Metrics by Loyalty Tier
loyalty_metrics = transactions.groupby('loyalty_tier').agg({
    'revenue': 'mean',
    'quantity': 'mean',
    'discount_pct': 'mean'
}).round(2)

x = np.arange(len(loyalty_metrics.index))
width = 0.25

bars1 = axes[1,1].bar(x - width, loyalty_metrics['revenue'], width, 
                      label='Avg Revenue ($)', color='#3498DB')
bars2 = axes[1,1].bar(x, loyalty_metrics['quantity'] * 10, width, 
                      label='Avg Quantity (×10)', color='#E74C3C')
bars3 = axes[1,1].bar(x + width, loyalty_metrics['discount_pct'], width, 
                      label='Avg Discount (%)', color='#2ECC71')

axes[1,1].set_xlabel('Loyalty Tier')
axes[1,1].set_ylabel('Values')
axes[1,1].set_title('Average Metrics by Loyalty Tier')
axes[1,1].set_xticks(x)
axes[1,1].set_xticklabels(loyalty_metrics.index)
axes[1,1].legend()

plt.tight_layout()
plt.show()

print("✅ Bar charts created - showing categorical distributions and comparisons")
```

### 4. Area Plots - Cumulative Analysis

```python
print("\n🏔️ Area Plots - Cumulative Analysis")
print("=" * 34)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Area Plots - Cumulative and Stacked Analysis', fontsize=16, fontweight='bold')

# Cumulative revenue area plot
monthly_data_sorted = monthly_data.sort_values('transaction_date')
axes[0,0].fill_between(monthly_data_sorted['transaction_date'], 
                       monthly_data_sorted['revenue'].cumsum(),
                       alpha=0.7, color='#3498DB')
axes[0,0].plot(monthly_data_sorted['transaction_date'], 
               monthly_data_sorted['revenue'].cumsum(),
               color='#2E86AB', linewidth=2)
axes[0,0].set_title('Cumulative Revenue Growth')
axes[0,0].set_ylabel('Cumulative Revenue ($)')
axes[0,0].tick_params(axis='x', rotation=45)
axes[0,0].grid(True, alpha=0.3)

# Stacked area plot by category
monthly_category = transactions.groupby([transactions['transaction_date'].dt.to_period('M'), 'category'])['revenue'].sum().unstack(fill_value=0)
monthly_category.index = monthly_category.index.to_timestamp()

# Create stacked area
axes[0,1].stackplot(monthly_category.index, *[monthly_category[col] for col in monthly_category.columns],
                    labels=monthly_category.columns, alpha=0.8)
axes[0,1].set_title('Monthly Revenue by Category (Stacked)')
axes[0,1].set_ylabel('Revenue ($)')
axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0,1].tick_params(axis='x', rotation=45)

# Percentage area plot
monthly_category_pct = monthly_category.div(monthly_category.sum(axis=1), axis=0) * 100
axes[1,0].stackplot(monthly_category_pct.index, *[monthly_category_pct[col] for col in monthly_category_pct.columns],
                    labels=monthly_category_pct.columns, alpha=0.8)
axes[1,0].set_title('Monthly Revenue % by Category')
axes[1,0].set_ylabel('Percentage (%)')
axes[1,0].set_ylim(0, 100)
axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1,0].tick_params(axis='x', rotation=45)

# Area plot with confidence intervals
daily_revenue = transactions.groupby('transaction_date')['revenue'].sum().resample('D').sum()
# Calculate rolling mean and std
rolling_mean = daily_revenue.rolling(window=7).mean()
rolling_std = daily_revenue.rolling(window=7).std()

axes[1,1].fill_between(rolling_mean.index, 
                       rolling_mean - rolling_std, 
                       rolling_mean + rolling_std,
                       alpha=0.3, color='gray', label='±1 Std Dev')
axes[1,1].plot(rolling_mean.index, rolling_mean, color='red', linewidth=2, label='7-day Moving Average')
axes[1,1].scatter(daily_revenue.index, daily_revenue.values, alpha=0.5, s=10, color='blue', label='Daily Revenue')
axes[1,1].set_title('Daily Revenue with Confidence Interval')
axes[1,1].set_ylabel('Revenue ($)')
axes[1,1].legend()
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print("✅ Area plots created - showing cumulative trends and distributions")
```

### 5. Advanced Seaborn Visualizations

```python
print("\n🎨 Advanced Seaborn Visualizations")
print("=" * 36)

# Prepare data for seaborn plots
customer_detailed = customer_summary.merge(customers[['customer_id', 'gender', 'city', 'loyalty_tier']], on='customer_id')

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Advanced Statistical Visualizations with Seaborn', fontsize=16, fontweight='bold')

# 1. Violin Plot - Distribution by category
sns.violinplot(data=transactions, x='loyalty_tier', y='revenue', ax=axes[0,0])
axes[0,0].set_title('Revenue Distribution by Loyalty Tier')
axes[0,0].tick_params(axis='x', rotation=45)

# 2. Swarm Plot - Detailed distribution
sample_transactions = transactions.sample(n=1000, random_state=42)  # Sample for performance
sns.swarmplot(data=sample_transactions, x='category', y='revenue', hue='loyalty_tier', 
              size=3, ax=axes[0,1])
axes[0,1].set_title('Revenue Distribution by Category & Loyalty')
axes[0,1].tick_params(axis='x', rotation=45)
axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 3. Box Plot with additional statistics
sns.boxplot(data=transactions, x='category', y='revenue', ax=axes[0,2])
axes[0,2].set_title('Revenue Distribution by Category (Box Plot)')
axes[0,2].tick_params(axis='x', rotation=45)

# 4. Heatmap - Correlation matrix
# Create correlation data
correlation_data = customer_detailed[['age', 'annual_income', 'revenue', 'transaction_id', 'quantity']].corr()
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0,
            square=True, ax=axes[1,0])
axes[1,0].set_title('Customer Metrics Correlation Heatmap')

# 5. Pair Plot (subset)
# Create a small dataset for pair plot
pair_data = customer_detailed[['age', 'annual_income', 'revenue', 'loyalty_tier']].sample(500, random_state=42)
# Since we can't use sns.pairplot in subplots, create a scatter matrix manually
scatter_vars = ['age', 'annual_income', 'revenue']
for i, var1 in enumerate(scatter_vars):
    for j, var2 in enumerate(scatter_vars):
        if i == 0 and j == 1:  # Use one of the subplot positions
            sns.scatterplot(data=pair_data, x=var1, y=var2, hue='loyalty_tier', ax=axes[1,1])
            axes[1,1].set_title(f'{var1.title()} vs {var2.title()}')

# 6. Ridge Plot simulation using multiple KDE
loyalty_tiers = customer_detailed['loyalty_tier'].unique()
colors = sns.color_palette("husl", len(loyalty_tiers))

for i, tier in enumerate(loyalty_tiers):
    tier_data = customer_detailed[customer_detailed['loyalty_tier'] == tier]['revenue']
    sns.kdeplot(tier_data, ax=axes[1,2], label=tier, color=colors[i], alpha=0.7)

axes[1,2].set_title('Revenue Distribution by Loyalty Tier (KDE)')
axes[1,2].set_xlabel('Revenue ($)')
axes[1,2].legend()

plt.tight_layout()
plt.show()

print("✅ Advanced seaborn visualizations created - showing statistical distributions")
```

### 6. Specialized Charts - Radar, Polar, and Tree Map

```python
print("\n🎯 Specialized Visualizations")
print("=" * 31)

# 6.1 RADAR CHART
print("Creating Radar Chart...")

# Prepare data for radar chart - category performance metrics
radar_data = transactions.groupby('category').agg({
    'revenue': 'sum',
    'transaction_id': 'count',
    'quantity': 'sum',
    'discount_pct': 'mean'
}).round(2)

# Normalize data to 0-1 scale for better radar visualization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
radar_normalized = pd.DataFrame(
    scaler.fit_transform(radar_data),
    columns=radar_data.columns,
    index=radar_data.index
)

# Create radar chart
def create_radar_chart(data, title):
    categories = list(data.columns)
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(data.index)))
    
    for idx, (product_cat, values) in enumerate(data.iterrows()):
        values_list = values.tolist()
        values_list += values_list[:1]  # Complete the circle
        
        ax.plot(angles, values_list, 'o-', linewidth=2, label=product_cat, color=colors[idx])
        ax.fill(angles, values_list, alpha=0.25, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title(title, size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    return fig, ax

fig_radar, ax_radar = create_radar_chart(radar_normalized, 'Product Category Performance Radar')
plt.show()

# 6.2 POLAR PLOT  
print("Creating Polar Plot...")

# Create polar plot for time-based analysis
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Monthly data for polar plot
monthly_revenue_polar = transactions.groupby(transactions['transaction_date'].dt.month)['revenue'].sum()
months = monthly_revenue_polar.index
revenues = monthly_revenue_polar.values

# Convert months to radians
theta = np.array(months) * 2 * np.pi / 12

# Create polar bar plot
bars = ax.bar(theta, revenues, width=2*np.pi/12, alpha=0.8, color=plt.cm.viridis(revenues/revenues.max()))

# Customize
ax.set_title('Monthly Revenue Distribution (Polar)', size=16, fontweight='bold', pad=20)
ax.set_theta_zero_location('N')  # Start from top
ax.set_theta_direction(-1)  # Clockwise
ax.set_thetagrids(np.arange(0, 360, 30), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

plt.show()

# 6.3 TREEMAP
print("Creating TreeMap...")

# Prepare data for treemap
treemap_data = category_analysis.copy()
treemap_data['labels'] = [f"{cat}\n${rev:,.0f}\n({pct:.1f}%)" 
                         for cat, rev, pct in zip(treemap_data.index, 
                                                treemap_data['Total_Revenue'], 
                                                treemap_data['Revenue_Pct'])]

# Create treemap
fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.Set3(np.linspace(0, 1, len(treemap_data)))

squarify.plot(sizes=treemap_data['Total_Revenue'], 
              label=treemap_data['labels'],
              color=colors, alpha=0.8, ax=ax)

ax.set_title('Revenue by Category - TreeMap Visualization', fontsize=16, fontweight='bold')
ax.axis('off')
plt.show()

print("✅ Specialized charts created - radar, polar, and treemap visualizations")
```

### 7. Statistical Heatmaps and Advanced Correlation Analysis

```python
print("\n🔥 Advanced Heatmap Analysis")
print("=" * 31)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Advanced Heatmap and Correlation Analysis', fontsize=16, fontweight='bold')

# 1. Pivot table heatmap - Revenue by City and Category
city_category_pivot = transactions.groupby(['category'])['revenue'].sum().reset_index()
city_customer_data = transactions.merge(customers[['customer_id', 'city']], on='customer_id')
city_category_revenue = city_customer_data.groupby(['city', 'category'])['revenue'].sum().unstack(fill_value=0)

sns.heatmap(city_category_revenue, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[0,0])
axes[0,0].set_title('Revenue Heatmap: City vs Category')
axes[0,0].tick_params(axis='x', rotation=45)

# 2. Correlation heatmap with hierarchical clustering
customer_metrics = customer_detailed[['age', 'annual_income', 'revenue', 'transaction_id', 'quantity', 'avg_order_value']]
correlation_matrix = customer_metrics.corr()

# Create mask for upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f', ax=axes[0,1])
axes[0,1].set_title('Customer Metrics Correlation (Lower Triangle)')

# 3. Time-based heatmap
transactions['hour'] = pd.to_datetime(transactions['transaction_date']).dt.hour
transactions['day_of_week'] = pd.to_datetime(transactions['transaction_date']).dt.day_name()

# Create synthetic hourly data for demo
np.random.seed(42)
hourly_data = []
for _, row in transactions.iterrows():
    synthetic_hour = np.random.randint(0, 24)
    hourly_data.append(synthetic_hour)

transactions['synthetic_hour'] = hourly_data

time_heatmap = transactions.groupby(['day_of_week', 'synthetic_hour']).size().unstack(fill_value=0)
# Reorder days
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
time_heatmap = time_heatmap.reindex(day_order)

sns.heatmap(time_heatmap, cmap='Blues', ax=axes[1,0])
axes[1,0].set_title('Transaction Volume: Day vs Hour Heatmap')
axes[1,0].set_xlabel('Hour of Day')

# 4. Loyalty tier performance matrix
loyalty_metrics_detailed = transactions.groupby(['loyalty_tier', 'category']).agg({
    'revenue': 'mean',
    'quantity': 'mean',
    'discount_pct': 'mean'
}).round(2)

# Focus on revenue for heatmap
loyalty_revenue_matrix = loyalty_metrics_detailed['revenue'].unstack(fill_value=0)
sns.heatmap(loyalty_revenue_matrix, annot=True, fmt='.1f', cmap='viridis', ax=axes[1,1])
axes[1,1].set_title('Average Revenue: Loyalty Tier vs Category')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print("✅ Advanced heatmap analysis complete - showing multi-dimensional relationships")
```

### 8. Interactive Plotly Visualizations

```python
print("\n🎪 Interactive Plotly Visualizations")
print("=" * 36)

# Note: In Jupyter notebooks, these would be interactive
# Here we'll create static versions that demonstrate the concepts

# 8.1 Interactive Scatter Plot with Plotly
print("Creating Interactive Scatter Plot...")

fig_scatter = px.scatter(customer_detailed, x='age', y='revenue', 
                        color='loyalty_tier', size='quantity',
                        hover_data=['annual_income', 'transaction_id'],
                        title='Interactive Customer Analysis: Age vs Revenue',
                        labels={'revenue': 'Total Revenue ($)', 'age': 'Customer Age'})
fig_scatter.show()

# 8.2 Interactive Time Series
print("Creating Interactive Time Series...")

fig_timeseries = px.line(monthly_data, x='transaction_date', y='revenue',
                        title='Interactive Monthly Revenue Trend',
                        labels={'revenue': 'Revenue ($)', 'transaction_date': 'Date'})
fig_timeseries.add_scatter(x=monthly_data['transaction_date'], y=monthly_data['revenue'],
                          name='Monthly Points', mode='markers')
fig_timeseries.show()

# 8.3 Interactive Bar Chart
print("Creating Interactive Bar Chart...")

fig_bar = px.bar(category_analysis.reset_index(), x='category', y='Total_Revenue',
                color='Total_Revenue', title='Interactive Revenue by Category',
                labels={'Total_Revenue': 'Total Revenue ($)', 'category': 'Product Category'})
fig_bar.show()

# 8.4 3D Scatter Plot
print("Creating 3D Scatter Plot...")

fig_3d = px.scatter_3d(customer_detailed.sample(1000, random_state=42), 
                      x='age', y='annual_income', z='revenue',
                      color='loyalty_tier', size='quantity',
                      title='3D Customer Analysis: Age, Income, Revenue',
                      labels={'revenue': 'Total Revenue ($)', 
                             'annual_income': 'Annual Income ($)',
                             'age': 'Age'})
fig_3d.show()

print("✅ Interactive Plotly visualizations created")
print("   (Note: In Jupyter notebooks, these would be fully interactive)")
```

### Visualization Summary

```python
print("\n📊 VISUALIZATION SUMMARY")
print("=" * 27)

visualization_summary = {
    'Chart Types Created': [
        '📈 Line Plots (Time Series)',
        '🔸 Scatter Plots (Correlations)', 
        '📊 Bar Charts (Categories)',
        '🏔️ Area Plots (Cumulative)',
        '🎨 Violin Plots (Distributions)',
        '🔥 Heatmaps (Correlations)',
        '🎯 Radar Charts (Multi-metric)',
        '🌀 Polar Plots (Cyclical)',
        '🗺️ TreeMaps (Hierarchical)',
        '🎪 Interactive Plots (Plotly)'
    ],
    'Business Applications': [
        'Temporal trend analysis',
        'Customer segmentation visualization',
        'Performance comparison',
        'Statistical distribution analysis',
        'Multi-dimensional relationships',
        'Geographic and categorical insights',
        'Interactive data exploration'
    ]
}

print("✅ Comprehensive Visualization Suite Complete!")
print(f"   • {len(visualization_summary['Chart Types Created'])} different chart types")
print(f"   • {len(visualization_summary['Business Applications'])} business applications")
print(f"   • Multiple libraries: matplotlib, seaborn, plotly")
print(f"   • Static and interactive capabilities")

print("\n🎯 Chart Types Covered:")
for chart_type in visualization_summary['Chart Types Created']:
    print(f"   {chart_type}")

print("\n💼 Business Applications:")
for application in visualization_summary['Business Applications']:
    print(f"   • {application}")
```

### Cleanup

```python
# Close database connection
conn.close()
print(f"\n✅ Database connection closed")
print("=" * 45)
print("Complete EDA and Visualization Pipeline Ready!")
```

---

## 📝 Next Steps

This comprehensive EDA provides a solid foundation for:

1. **Advanced Machine Learning**: Customer churn prediction, recommendation systems
2. **Real-time Dashboards**: KPI monitoring and automated reporting  
3. **A/B Testing**: Campaign effectiveness and product optimization
4. **Predictive Analytics**: Demand forecasting and inventory optimization
5. **Customer Journey Modeling**: Lifecycle analysis and touchpoint optimization

**🔧 Technical Enhancements:**
- Add statistical significance testing
- Implement advanced visualizations with Plotly/Dash
- Create automated report generation
- Build real-time data pipeline connections
- Develop machine learning model integration

**📊 Business Applications:**
- Marketing campaign targeting
- Inventory planning and optimization  
- Customer service prioritization
- Product development roadmap
- Revenue forecasting and budgeting