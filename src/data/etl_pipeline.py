# This is our ETL (Extract, Transform, Load) pipeline - it cleans and organizes our data!

import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine
from database import init_database, Customer, Transaction, CustomerRFM
import logging
import os

# Set up logging so we can see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ETLPipeline:
    def __init__(self):
        self.engine = init_database()
        logger.info(" ETL Pipeline initialized!")
    
    def extract(self):
        """Step 1: Extract data from our CSV files"""
        logger.info(" Extracting data from source files...")
        
        try:
            customers = pd.read_csv('data/raw/customers.csv', parse_dates=['signup_date'])
            transactions = pd.read_csv('data/raw/transactions.csv', parse_dates=['transaction_date'])
            
            logger.info(f" Extracted {len(customers)} customers and {len(transactions)} transactions")
            return customers, transactions
            
        except Exception as e:
            logger.error(f" Error extracting data: {e}")
            raise
    
    def transform(self, customers, transactions):
        """Step 2: Clean and transform our data"""
        logger.info(" Transforming and cleaning data...")
        
        # Clean customers data - remove duplicates
        customers_clean = customers.drop_duplicates('customer_id')
        logger.info(f" Cleaned customers: {len(customers_clean)} unique customers")
        
        # Clean transactions - only keep completed transactions with positive amounts
        transactions_clean = transactions[
            (transactions['status'] == 'Completed') & 
            (transactions['total_amount'] > 0)
        ].copy()
        logger.info(f" Cleaned transactions: {len(transactions_clean)} valid transactions")
        
        # Calculate RFM metrics (Recency, Frequency, Monetary)
        logger.info(" Calculating RFM metrics...")
        rfm_data = self._calculate_rfm(transactions_clean)
        
        return customers_clean, transactions_clean, rfm_data
    
    def _calculate_rfm(self, transactions):
        """Calculate Recency, Frequency, Monetary values for each customer"""
        
        # Find the most recent transaction date
        max_date = transactions['transaction_date'].max()
        logger.info(f" Most recent transaction: {max_date}")
        
        # Group by customer and calculate metrics
        rfm = transactions.groupby('customer_id').agg({
            'transaction_date': lambda x: (max_date - x.max()).days,  # Recency: days since last purchase
            'transaction_id': 'count',                                # Frequency: number of purchases
            'total_amount': 'sum'                                     # Monetary: total spending
        }).reset_index()
        
        rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
        
        # Create RFM scores (1-5, where 5 is best)
        logger.info(" Creating RFM scores...")
        
        # Recency: lower recency (more recent) is better → score 5
        rfm['r_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
        
        # Frequency: higher frequency is better → score 5
        rfm['f_score'] = pd.qcut(rfm['frequency'], 5, labels=[1, 2, 3, 4, 5])
        
        # Monetary: higher spending is better → score 5
        rfm['m_score'] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
        
        # Combine scores
        rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
        
        # Create customer segments
        rfm['segment'] = rfm.apply(self._segment_customer, axis=1)
        
        logger.info(f" RFM calculation complete! Created segments for {len(rfm)} customers")
        return rfm
    
    def _segment_customer(self, row):
        """Assign customer segment based on RFM scores"""
        r, f, m = row['r_score'], row['f_score'], row['m_score']
        
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif r >= 4 and f >= 3:
            return 'Loyal Customers'
        elif r >= 3:
            return 'Potential Loyalists'
        elif r >= 2:
            return 'At Risk'
        else:
            return 'Cannot Lose'
    
    def load(self, customers, transactions, rfm_data):
        """Step 3: Load data into our database"""
        logger.info(" Loading data into database...")
        
        try:
            # Load customers table
            customers.to_sql('customers', self.engine, if_exists='replace', index=False)
            logger.info(f" Loaded {len(customers)} customers")
            
            # Load transactions table
            transactions.to_sql('transactions', self.engine, if_exists='replace', index=False)
            logger.info(f" Loaded {len(transactions)} transactions")
            
            # Load RFM data
            rfm_data[['customer_id', 'recency', 'frequency', 'monetary', 'rfm_score', 'segment']].to_sql(
                'customer_rfm', self.engine, if_exists='replace', index=False
            )
            logger.info(f" Loaded RFM data for {len(rfm_data)} customers")
            
        except Exception as e:
            logger.error(f" Error loading data: {e}")
            raise
    
    def run(self):
        """Execute the complete ETL pipeline"""
        logger.info(" Starting ETL pipeline...")
        
        try:
            # Extract
            customers, transactions = self.extract()
            
            # Transform
            customers_clean, transactions_clean, rfm_data = self.transform(customers, transactions)
            
            # Load
            self.load(customers_clean, transactions_clean, rfm_data)
            
            logger.info(" ETL pipeline completed successfully!")
            
            # Print summary
            self._print_summary(rfm_data)
            
        except Exception as e:
            logger.error(f" ETL pipeline failed: {e}")
            raise
    
    def _print_summary(self, rfm_data):
        """Print a nice summary of our data"""
        print("\n" + "="*50)
        print(" ETL PIPELINE SUMMARY")
        print("="*50)
        
        # Customer segments distribution
        segment_counts = rfm_data['segment'].value_counts()
        print("\n👥 Customer Segments:")
        for segment, count in segment_counts.items():
            percentage = (count / len(rfm_data)) * 100
            print(f"   {segment}: {count} customers ({percentage:.1f}%)")
        
        # RFM statistics
        print(f"\n RFM Statistics:")
        print(f"   Average Recency: {rfm_data['recency'].mean():.1f} days")
        print(f"   Average Frequency: {rfm_data['frequency'].mean():.1f} purchases")
        print(f"   Average Monetary: ${rfm_data['monetary'].mean():.2f}")
        
        print(f"\n Database file: data/customer_analytics.db")
        print("="*50)

# Let's make it easy to run our pipeline!
if __name__ == "__main__":
    print("Starting Customer Analytics ETL Pipeline...")
    print("This will:")
    print("1.  Extract data from CSV files")
    print("2.  Clean and transform the data") 
    print("3.  Calculate customer segments (RFM)")
    print("4.  Load everything into a database")
    print("\n" + "─" * 50)
    
    pipeline = ETLPipeline()
    pipeline.run()