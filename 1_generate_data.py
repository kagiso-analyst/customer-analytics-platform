import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

print("Welcome to your Data Science Project!")
print("Let's create some customer data...")

# The data folder if it doesn't exist
os.makedirs('data/raw', exist_ok=True)

def generate_customer_data(n_customers=1000):
    """Create fake customer data"""
    print(f"Creating {n_customers} customers...")
    
    customers = []
    for i in range(n_customers):
        join_date = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1095))
        customers.append({
            'customer_id': f'CUST_{i:06d}',
            'name': f'Customer_{i}',
            'email': f'customer_{i}@example.com',
            'signup_date': join_date,
            'country': random.choice(['US', 'UK', 'Germany', 'France', 'Canada', 'Australia']),
            'age_group': random.choice(['18-25', '26-35', '36-45', '46-55', '55+']),
            'acquisition_channel': random.choice(['Organic', 'Paid Social', 'Email', 'Referral', 'Direct'])
        })
    
    return pd.DataFrame(customers)

def generate_transaction_data(customers_df, n_transactions=50000):
    """Create fake transaction data"""
    print(f"Creating {n_transactions} transactions...")
    
    transactions = []
    products = [
        {'id': 'P001', 'name': 'Laptop', 'category': 'Electronics', 'price_range': 'high'},
        {'id': 'P002', 'name': 'Smartphone', 'category': 'Electronics', 'price_range': 'high'},
        {'id': 'P003', 'name': 'Headphones', 'category': 'Electronics', 'price_range': 'medium'},
        {'id': 'P004', 'name': 'Book', 'category': 'Education', 'price_range': 'low'},
        {'id': 'P005', 'name': 'T-Shirt', 'category': 'Clothing', 'price_range': 'low'},
    ]
    
    for i in range(n_transactions):
        customer = random.choice(customers_df['customer_id'].values)
        product = random.choice(products)
        
        #Base price based on product
        base_price = {
            'high': random.uniform(800, 2000),
            'medium': random.uniform(100, 800),
            'low': random.uniform(10, 100)
        }[product['price_range']]
        
        price = base_price * random.uniform(0.8, 1.2)
        quantity = random.randint(1, 3)
        
        #Create transaction date
        customer_data = customers_df[customers_df['customer_id'] == customer].iloc[0]
        signup_date = customer_data['signup_date']
        max_date = min(datetime.now(), signup_date + timedelta(days=365*3))
        
        if max_date > signup_date:
            days_after_signup = random.randint(0, (max_date - signup_date).days)
            transaction_date = signup_date + timedelta(days=days_after_signup)
            
            transactions.append({
                'transaction_id': f'TXN_{i:08d}',
                'customer_id': customer,
                'product_id': product['id'],
                'product_name': product['name'],
                'category': product['category'],
                'quantity': quantity,
                'unit_price': round(price, 2),
                'total_amount': round(price * quantity, 2),
                'transaction_date': transaction_date,
                'payment_method': random.choice(['Credit Card', 'PayPal', 'Bank Transfer']),
                'status': random.choices(['Completed', 'Refunded'], weights=[0.95, 0.05])[0]
            })
    
    return pd.DataFrame(transactions)

# Run it
if __name__ == "__main__":
    print("Starting data generation...")
    
    customers_df = generate_customer_data(1000)
    transactions_df = generate_transaction_data(customers_df, 50000)
    
    # Save to CSV files
    customers_df.to_csv('data/raw/customers.csv', index=False)
    transactions_df.to_csv('data/raw/transactions.csv', index=False)
    
    print("Data generation complete!")
    print(f"Created {len(customers_df)} customers")
    print(f"Created {len(transactions_df)} transactions")
    print("Saved to: data/raw/customers.csv and data/raw/transactions.csv")
    
    # Show a little preview
    print("\n Customer data preview:")
    print(customers_df.head(3))
    print("\n Transaction data preview:")
    print(transactions_df.head(3))