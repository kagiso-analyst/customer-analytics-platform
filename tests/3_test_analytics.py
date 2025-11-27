# Test script for Phase 3 - Machine Learning & Analytics

import sys
import os
from sqlalchemy import create_engine

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from analytics.customer_segmentation import CustomerSegmentation
from models.clv_prediction import CLVPredictor

def test_phase_3():
    print(" PHASE 3 TEST: Machine Learning & Advanced Analytics")
    print("=" * 60)
    
    # Connect to database
    engine = create_engine('sqlite:///data/customer_analytics.db')
    
    print("1. Testing Customer Segmentation...")
    try:
        segmentation = CustomerSegmentation(engine)
        clustered_data, kmeans, scaler = segmentation.perform_kmeans_clustering(n_clusters=5)
        clustered_data, summary, names = segmentation.analyze_clusters(clustered_data)
        print("    Customer Segmentation: SUCCESS")
    except Exception as e:
        print(f"    Customer Segmentation: FAILED - {e}")
        return
    
    print("\n2. Testing CLV Prediction...")
    try:
        clv_predictor = CLVPredictor(engine)
        results, feature_importance = clv_predictor.train_model()
        print("    CLV Prediction: SUCCESS")
    except Exception as e:
        print(f"    CLV Prediction: FAILED - {e}")
        return
    
    print("\n3. Testing Visualizations...")
    try:
        # Test segmentation visualizations
        fig_pie, fig_3d, fig_comparison = segmentation.create_segmentation_visualizations(clustered_data)
        print("    Segmentation Visualizations: SUCCESS")
        
        # Test CLV visualizations
        data = clv_predictor.prepare_features()
        fig_dist, fig_channel, fig_segment = clv_predictor.create_clv_visualizations(data)
        print("    CLV Visualizations: SUCCESS")
    except Exception as e:
        print(f"    Visualizations: FAILED - {e}")
        return
    
    print("\n4. Generating Reports...")
    try:
        segmentation.generate_segmentation_report(clustered_data, summary, names)
        clv_predictor.generate_clv_report(data, results)
        print("    Reports: SUCCESS")
    except Exception as e:
        print(f"    Reports: FAILED - {e}")
        return
    
    print("\n" + "=" * 60)
    print(" PHASE 3 TEST COMPLETED SUCCESSFULLY!")
    print("\n What we've built:")
    print("   •  Advanced customer segmentation with K-means clustering")
    print("   •  Customer Lifetime Value prediction with multiple ML models")
    print("   •  Interactive visualizations with Plotly")
    print("   •  Comprehensive business intelligence reports")
    print("   •  Actionable business recommendations")
    
    print("\n Ready for Phase 4: Interactive Dashboard!")
    print("   We'll build a beautiful web dashboard to showcase all this work!")

if __name__ == "__main__":
    test_phase_3()