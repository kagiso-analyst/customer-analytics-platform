# This script tests that our database is working correctly!

import pandas as pd
from sqlalchemy import create_engine, inspect
import os

print(" Testing our database setup...")

def test_database():
    # Connect to our database
    engine = create_engine('sqlite:///data/customer_analytics.db')
    
    # Check if database file exists
    if not os.path.exists('data/customer_analytics.db'):
        print(" Database file not found! Run the ETL pipeline first.")
        return
    
    # Check what tables we have
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    print(f" Found {len(tables)} tables: {tables}")
    
    # Show some data from each table
    for table in tables:
        print(f"\n📊 Table: {table}")
        try:
            # Read first few rows
            df = pd.read_sql(f"SELECT * FROM {table} LIMIT 3", engine)
            print(f"   Columns: {list(df.columns)}")
            print(f"   Sample data:")
            for _, row in df.iterrows():
                print(f"     {dict(row)}")
        except Exception as e:
            print(f"    Error reading table: {e}")
    
    # Show some statistics
    print("\n📈 Database Statistics:")
    for table in tables:
        count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table}", engine)['count'].iloc[0]
        print(f"   {table}: {count} rows")
    
    print("\n Database test complete!")

if __name__ == "__main__":
    test_database()