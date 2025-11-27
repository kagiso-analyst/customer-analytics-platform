# This file sets up our database structure - like building shelves to organize our data!

from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

print(" Setting up database structure...")

# This is the foundation for our database tables
Base = declarative_base()

class Customer(Base):
    __tablename__ = 'customers'
    
    customer_id = Column(String(50), primary_key=True)
    name = Column(String(100))
    email = Column(String(100))
    signup_date = Column(DateTime)
    country = Column(String(50))
    age_group = Column(String(20))
    acquisition_channel = Column(String(50))
    
    def __repr__(self):
        return f"<Customer(id={self.customer_id}, name={self.name})>"

class Transaction(Base):
    __tablename__ = 'transactions'
    
    transaction_id = Column(String(50), primary_key=True)
    customer_id = Column(String(50))
    product_id = Column(String(50))
    product_name = Column(String(100))
    category = Column(String(50))
    quantity = Column(Integer)
    unit_price = Column(Float)
    total_amount = Column(Float)
    transaction_date = Column(DateTime)
    payment_method = Column(String(50))
    status = Column(String(20))
    
    def __repr__(self):
        return f"<Transaction(id={self.transaction_id}, amount={self.total_amount})>"

class CustomerRFM(Base):
    __tablename__ = 'customer_rfm'
    
    customer_id = Column(String(50), primary_key=True)
    recency = Column(Integer)  # Days since last purchase
    frequency = Column(Integer)  # Number of purchases
    monetary = Column(Float)    # Total spending
    rfm_score = Column(String(10))
    segment = Column(String(50))
    
    def __repr__(self):
        return f"<CustomerRFM(id={self.customer_id}, segment={self.segment})>"

def init_database():
    """Create our database and all tables"""
    # Using SQLite (a simple file-based database)
    engine = create_engine('sqlite:///data/customer_analytics.db')
    
    print(" Creating database tables...")
    Base.metadata.create_all(engine)
    print(" Database tables created successfully!")
    
    return engine

# Let's test our database setup
if __name__ == "__main__":
    engine = init_database()
    
    # Check what tables we created
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f" Created tables: {tables}")