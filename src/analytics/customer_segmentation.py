# This file performs advanced customer segmentation using machine learning!

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns

print("👥 Starting advanced customer segmentation...")

class CustomerSegmentation:
    def __init__(self, engine):
        self.engine = engine
        self.scaler = StandardScaler()
        print(" Customer Segmentation initialized!")
    
    def get_customer_data(self):
        """Get enriched customer data with RFM and behavioral features"""
        print(" Loading customer data from database...")
        
        query = """
        SELECT 
            c.*,
            r.recency,
            r.frequency,
            r.monetary,
            r.segment as rfm_segment,
            COUNT(DISTINCT t.category) as unique_categories,
            AVG(t.total_amount) as avg_transaction_value,
            MAX(t.transaction_date) as last_purchase_date,
            COUNT(DISTINCT t.product_id) as unique_products,
            (JULIANDAY('now') - JULIANDAY(c.signup_date)) as days_since_signup
        FROM customers c
        LEFT JOIN customer_rfm r ON c.customer_id = r.customer_id
        LEFT JOIN transactions t ON c.customer_id = t.customer_id
        GROUP BY c.customer_id
        """
        
        data = pd.read_sql(query, self.engine)
        print(f" Loaded {len(data)} customers with enriched features")
        return data
    
    def perform_kmeans_clustering(self, n_clusters=5):
        """Perform K-means clustering on customer data"""
        print(f" Performing K-means clustering with {n_clusters} clusters...")
        
        data = self.get_customer_data()
        
        # Prepare features for clustering
        features = data[['recency', 'frequency', 'monetary', 'unique_categories', 'avg_transaction_value']].fillna(0)
        
        print(" Features for clustering:")
        for feature in features.columns:
            print(f"   - {feature}: mean={features[feature].mean():.2f}")
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_scaled)
        
        # Calculate silhouette score (how well separated clusters are)
        silhouette_avg = silhouette_score(features_scaled, clusters)
        print(f" Silhouette Score: {silhouette_avg:.3f} (higher is better)")
        
        # Add clusters to data
        data['cluster'] = clusters
        data['cluster'] = 'Cluster_' + data['cluster'].astype(str)
        
        print(" Clustering completed successfully!")
        return data, kmeans, self.scaler
    
    def analyze_clusters(self, clustered_data):
        """Analyze and describe each cluster"""
        print(" Analyzing cluster characteristics...")
        
        cluster_summary = clustered_data.groupby('cluster').agg({
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean',
            'unique_categories': 'mean',
            'avg_transaction_value': 'mean',
            'customer_id': 'count'
        }).round(2)
        
        cluster_summary['size_percentage'] = (cluster_summary['customer_id'] / len(clustered_data) * 100).round(1)
        cluster_summary = cluster_summary.rename(columns={'customer_id': 'customer_count'})
        
        # Name clusters based on characteristics
        cluster_names = {}
        for cluster_id in cluster_summary.index:
            recency = cluster_summary.loc[cluster_id, 'recency']
            frequency = cluster_summary.loc[cluster_id, 'frequency']
            monetary = cluster_summary.loc[cluster_id, 'monetary']
            
            if recency < 30 and frequency > 10 and monetary > 1000:
                cluster_names[cluster_id] = 'VIP Customers'
            elif recency < 60 and frequency > 5:
                cluster_names[cluster_id] = 'Loyal Customers'
            elif recency > 180:
                cluster_names[cluster_id] = 'At Risk Customers'
            elif frequency == 1:
                cluster_names[cluster_id] = 'One-time Buyers'
            else:
                cluster_names[cluster_id] = 'Regular Customers'
        
        clustered_data['cluster_name'] = clustered_data['cluster'].map(cluster_names)
        
        print(" Cluster analysis complete!")
        return clustered_data, cluster_summary, cluster_names
    
    def create_segmentation_visualizations(self, clustered_data):
        """Create interactive visualizations for customer segments"""
        print(" Creating visualizations...")
        
        # 1. Cluster distribution pie chart
        cluster_dist = clustered_data['cluster_name'].value_counts()
        fig_pie = px.pie(
            values=cluster_dist.values,
            names=cluster_dist.index,
            title="Customer Segment Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        # 2. 3D Scatter plot of clusters
        fig_3d = px.scatter_3d(
            clustered_data, 
            x='recency', 
            y='frequency', 
            z='monetary',
            color='cluster_name',
            title='Customer Segments - RFM 3D Visualization',
            hover_data=['customer_id', 'country', 'age_group'],
            size='avg_transaction_value',
            opacity=0.7
        )
        
        # 3. Cluster comparison bar chart
        cluster_means = clustered_data.groupby('cluster_name').agg({
            'recency': 'mean',
            'frequency': 'mean', 
            'monetary': 'mean',
            'avg_transaction_value': 'mean'
        }).reset_index()
        
        # Create subplots for comparison
        fig_comparison = go.Figure()
        
        metrics = ['recency', 'frequency', 'monetary', 'avg_transaction_value']
        metric_names = ['Recency (days)', 'Frequency', 'Monetary ($)', 'Avg Transaction ($)']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            fig_comparison.add_trace(go.Bar(
                name=name,
                x=cluster_means['cluster_name'],
                y=cluster_means[metric],
                text=cluster_means[metric].round(1),
                visible=(i == 0)  # Only show first metric initially
            ))
        
        # Add dropdown menu
        fig_comparison.update_layout(
            title='Cluster Characteristics Comparison',
            xaxis_title='Customer Segment',
            yaxis_title='Value',
            updatemenus=[{
                'buttons': [
                    {
                        'label': metric_names[i],
                        'method': 'update',
                        'args': [{'visible': [j == i for j in range(len(metrics))]},
                               {'yaxis.title': metric_names[i]}]
                    } for i in range(len(metrics))
                ],
                'direction': 'down',
                'showactive': True,
            }]
        )
        
        print(" Visualizations created!")
        return fig_pie, fig_3d, fig_comparison
    
    def generate_segmentation_report(self, clustered_data, cluster_summary, cluster_names):
        """Generate a comprehensive segmentation report"""
        print("\n" + "="*60)
        print(" CUSTOMER SEGMENTATION REPORT")
        print("="*60)
        
        print(f"\n Overall Statistics:")
        print(f"   Total Customers: {len(clustered_data)}")
        print(f"   Total Segments: {len(cluster_summary)}")
        
        print(f"\n👥 Segment Breakdown:")
        for cluster_id, row in cluster_summary.iterrows():
            segment_name = cluster_names[cluster_id]
            print(f"\n   {segment_name}:")
            print(f"     • Customers: {row['customer_count']} ({row['size_percentage']}%)")
            print(f"     • Avg Recency: {row['recency']} days")
            print(f"     • Avg Frequency: {row['frequency']} purchases")
            print(f"     • Avg Spending: ${row['monetary']:,.2f}")
            print(f"     • Avg Transaction: ${row['avg_transaction_value']:.2f}")
        
        # Business recommendations
        print(f"\n Business Recommendations:")
        for cluster_id, segment_name in cluster_names.items():
            cluster_data = clustered_data[clustered_data['cluster_name'] == segment_name]
            
            if segment_name == 'VIP Customers':
                print(f"    {segment_name}: Offer exclusive rewards and early access to new products")
            elif segment_name == 'Loyal Customers':
                print(f"    {segment_name}: Implement loyalty program and personalized offers")
            elif segment_name == 'At Risk Customers':
                print(f"     {segment_name}: Run win-back campaigns with special discounts")
            elif segment_name == 'One-time Buyers':
                print(f"    {segment_name}: Send follow-up offers and educational content")
            else:
                print(f"    {segment_name}: Encourage repeat purchases with bundle deals")

# Let's test our segmentation!
if __name__ == "__main__":
    print(" Testing Customer Segmentation...")
    
    # Connect to database
    engine = create_engine('sqlite:///data/customer_analytics.db')
    segmentation = CustomerSegmentation(engine)
    
    # Perform clustering
    clustered_data, kmeans, scaler = segmentation.perform_kmeans_clustering(n_clusters=5)
    
    # Analyze clusters
    clustered_data, summary, names = segmentation.analyze_clusters(clustered_data)
    
    # Create visualizations
    fig_pie, fig_3d, fig_comparison = segmentation.create_segmentation_visualizations(clustered_data)
    
    # Generate report
    segmentation.generate_segmentation_report(clustered_data, summary, names)
    
    print("\n Customer segmentation testing complete!")
    print(" To save visualizations, use: fig_pie.write_html('segmentation_pie.html')")