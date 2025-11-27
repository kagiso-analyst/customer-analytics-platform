#  Dashboard Runner - Easy way to start your dashboard!

import subprocess
import sys
import os
import pandas as pd
from sqlalchemy import create_engine

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = ['streamlit', 'plotly', 'sqlalchemy', 'pandas', 'scikit-learn']
    
    print(" Checking dependencies...")
    for package in required_packages:
        try:
            __import__(package)
            print(f"    {package}")
        except ImportError:
            print(f"    {package} - Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print(" All dependencies checked!")

def check_data():
    """Check if data exists and is ready"""
    print("\n Checking data...")
    
    if not os.path.exists('data/customer_analytics.db'):
        print("    Database not found!")
        print("    Please run the ETL pipeline first:")
        print("    python src/data/etl_pipeline.py")
        return False
    
    try:
        engine = create_engine('sqlite:///data/customer_analytics.db')
        # Test connection
        pd.read_sql("SELECT 1", engine)
        print("   Database connection successful")
        return True
    except Exception as e:
        print(f"    Database error: {e}")
        return False

def main():
    print(" Customer Analytics Dashboard Runner")
    print("=" * 50)
    
    # Check dependencies
    check_dependencies()
    
    # Check data
    if not check_data():
        print("\n Please fix the data issues above before running the dashboard.")
        return
    
    print("\n Starting Dashboard...")
    print("   📊 Dashboard will open in your browser")
    print("   ⏹️  Press Ctrl+C to stop the dashboard")
    print("   🌐 Access at: http://localhost:8501")
    print("\n" + "=" * 50)
    
    try:
        # Run Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n Dashboard stopped. Goodbye!")
    except Exception as e:
        print(f"\n Error starting dashboard: {e}")

if __name__ == "__main__":
    main()