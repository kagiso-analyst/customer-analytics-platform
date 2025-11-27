# MAIN DASHBOARD - Streamlit Cloud Compatible Version

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Configure the page
st.set_page_config(
    page_title="Customer Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .segment-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitCustomerAnalytics:
    def __init__(self):
        self.customer_data = self.generate_sample_data()
    
    @st.cache_data(ttl=3600)
    def generate_sample_data(_self):
        """Generate realistic sample data for Streamlit Cloud"""
        np.random.seed(42)
        
        # Generate customer data
        n_customers = 1000
        
        customers = []
        for i in range(n_customers):
            customers.append({
                'customer_id': f'CUST_{i:06d}',
                'country': np.random.choice(['US', 'UK', 'Germany', 'France', 'Canada', 'Australia']),
                'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '55+']),
                'acquisition_channel': np.random.choice(['Organic', 'Paid Social', 'Email', 'Referral', 'Direct']),
                'recency': np.random.randint(1, 365),  # days since last purchase
                'frequency': np.random.randint(1, 100),  # number of purchases
                'monetary': np.random.uniform(100, 10000),  # total spending
                'rfm_segment': np.random.choice(['Champions', 'Loyal Customers', 'Potential Loyalists', 'At Risk', 'Cannot Lose'], 
                                              p=[0.1, 0.2, 0.3, 0.3, 0.1])
            })
        
        df = pd.DataFrame(customers)
        
        # Add some realistic patterns
        df.loc[df['rfm_segment'] == 'Champions', 'monetary'] *= 3
        df.loc[df['rfm_segment'] == 'Loyal Customers', 'monetary'] *= 2
        df.loc[df['rfm_segment'] == 'At Risk', 'recency'] = np.random.randint(180, 365, 
                                                                             size=(df['rfm_segment'] == 'At Risk').sum())
        
        return df
    
    def perform_clustering(self, n_clusters=5):
        """Perform simple customer segmentation"""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        features = self.customer_data[['recency', 'frequency', 'monetary']].copy()
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_scaled)
        
        self.customer_data['cluster'] = clusters
        self.customer_data['cluster_name'] = self.customer_data['cluster'].map({
            0: 'VIP Customers',
            1: 'Loyal Customers', 
            2: 'Regular Customers',
            3: 'At Risk Customers',
            4: 'One-time Buyers'
        })
        
        return kmeans, scaler
    
    def calculate_clv(self):
        """Calculate simple Customer Lifetime Value"""
        # Simple CLV calculation based on RFM
        self.customer_data['clv_score'] = (
            self.customer_data['monetary'] * 
            self.customer_data['frequency'] / 
            (self.customer_data['recency'] + 1)  # +1 to avoid division by zero
        )
        
        # Normalize CLV score
        self.customer_data['clv_score'] = (
            self.customer_data['clv_score'] / self.customer_data['clv_score'].max() * 100
        )
        
        return self.customer_data
    
    def setup_sidebar(self):
        """Create the sidebar with filters and controls"""
        st.sidebar.title("🎛️ Dashboard Controls")
        
        st.sidebar.subheader("👥 Customer Segments")
        segments = st.sidebar.multiselect(
            "Select segments to display:",
            ["All", "Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Cannot Lose"],
            default=["All"]
        )
        
        st.sidebar.subheader("🌍 Geography")
        countries = st.sidebar.multiselect(
            "Select countries:",
            ["All", "US", "UK", "Germany", "France", "Canada", "Australia"],
            default=["All"]
        )
        
        return {
            'segments': segments,
            'countries': countries
        }
    
    def show_header(self):
        """Display the main header"""
        st.markdown('<h1 class="main-header">📊 Customer Analytics Platform</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        **Real-time customer intelligence platform** providing actionable insights into 
        customer behavior, lifetime value, and retention strategies.
        
        *Note: This demo uses generated sample data. In production, this would connect to your live database.*
        """)
        
        st.markdown("<br>", unsafe_allow_html=True)
    
    def show_kpi_metrics(self):
        """Display key performance indicators"""
        st.header("📈 Key Performance Indicators")
        
        # Calculate metrics
        total_customers = len(self.customer_data)
        total_revenue = self.customer_data['monetary'].sum()
        avg_clv = self.customer_data['monetary'].mean()
        avg_frequency = self.customer_data['frequency'].mean()
        repeat_rate = (self.customer_data['frequency'] > 1).mean() * 100
        
        # Create columns for metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Customers", f"{total_customers:,}")
        
        with col2:
            st.metric("Total Revenue", f"${total_revenue:,.0f}")
        
        with col3:
            st.metric("Avg Customer Value", f"${avg_clv:,.0f}")
        
        with col4:
            st.metric("Avg Purchase Frequency", f"{avg_frequency:.1f}")
        
        with col5:
            st.metric("Repeat Purchase Rate", f"{repeat_rate:.1f}%")
        
        st.markdown("<br>", unsafe_allow_html=True)
    
    def show_customer_segmentation(self):
        """Display customer segmentation analysis"""
        st.header("👥 Customer Segmentation Analysis")
        
        with st.spinner("Analyzing customer segments..."):
            # Perform clustering
            kmeans, scaler = self.perform_clustering()
            
            # Segment distribution
            col1, col2 = st.columns(2)
            
            with col1:
                segment_dist = self.customer_data['rfm_segment'].value_counts()
                fig_segment = px.pie(
                    values=segment_dist.values,
                    names=segment_dist.index,
                    title="Customer Segment Distribution (RFM)",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_segment.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_segment, use_container_width=True)
            
            with col2:
                # Spending by segment
                fig_spending = px.box(
                    self.customer_data,
                    x='rfm_segment',
                    y='monetary',
                    title="Spending Distribution by RFM Segment",
                    color='rfm_segment'
                )
                fig_spending.update_layout(showlegend=False)
                st.plotly_chart(fig_spending, use_container_width=True)
            
            # ML Clustering Results
            st.subheader("🤖 ML-Powered Customer Clustering")
            cluster_dist = self.customer_data['cluster_name'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_cluster = px.bar(
                    x=cluster_dist.index,
                    y=cluster_dist.values,
                    title="ML Cluster Distribution",
                    labels={'x': 'Cluster', 'y': 'Number of Customers'},
                    color=cluster_dist.values,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_cluster, use_container_width=True)
            
            with col2:
                # 3D Scatter plot of clusters
                fig_3d = px.scatter_3d(
                    self.customer_data,
                    x='recency',
                    y='frequency', 
                    z='monetary',
                    color='cluster_name',
                    title='Customer Segments - RFM 3D Visualization',
                    hover_data=['country', 'age_group'],
                    size='monetary',
                    opacity=0.7
                )
                st.plotly_chart(fig_3d, use_container_width=True)
            
            # Segment insights
            st.subheader("📋 Segment Insights")
            
            for segment in ['VIP Customers', 'Loyal Customers', 'At Risk Customers']:
                segment_data = self.customer_data[self.customer_data['cluster_name'] == segment]
                
                with st.expander(f"🔍 {segment} - {len(segment_data)} customers"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Avg Recency", f"{segment_data['recency'].mean():.0f} days")
                    with col2:
                        st.metric("Avg Frequency", f"{segment_data['frequency'].mean():.1f}")
                    with col3:
                        st.metric("Avg Spending", f"${segment_data['monetary'].mean():,.0f}")
                    
                    # Recommendations
                    if segment == 'VIP Customers':
                        st.info("**🎯 Strategy:** Offer exclusive rewards, early access, and premium support")
                    elif segment == 'Loyal Customers':
                        st.info("**🎯 Strategy:** Strengthen with loyalty programs and personalized offers")
                    elif segment == 'At Risk Customers':
                        st.warning("**🎯 Strategy:** Implement win-back campaigns with special discounts")
    
    def show_clv_analysis(self):
        """Display Customer Lifetime Value analysis"""
        st.header("💰 Customer Lifetime Value Analysis")
        
        with st.spinner("Calculating customer lifetime values..."):
            # Calculate CLV
            self.calculate_clv()
            
            # CLV Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_predicted_value = self.customer_data['monetary'].sum() * 0.1  # Simulated future value
                st.metric("Total Predicted Value", f"${total_predicted_value:,.0f}")
            
            with col2:
                avg_clv = self.customer_data['clv_score'].mean()
                st.metric("Average CLV Score", f"{avg_clv:.1f}/100")
            
            with col3:
                high_value_threshold = self.customer_data['clv_score'].quantile(0.8)
                st.metric("High-Value Threshold", f"{high_value_threshold:.1f}")
            
            with col4:
                high_value_customers = (self.customer_data['clv_score'] >= high_value_threshold).sum()
                st.metric("High-Value Customers", f"{high_value_customers}")
            
            # CLV Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                fig_clv_dist = px.histogram(
                    self.customer_data,
                    x='clv_score',
                    nbins=20,
                    title="Customer Lifetime Value Distribution",
                    labels={'clv_score': 'CLV Score'},
                    color_discrete_sequence=['#636EFA']
                )
                st.plotly_chart(fig_clv_dist, use_container_width=True)
            
            with col2:
                clv_by_channel = self.customer_data.groupby('acquisition_channel')['clv_score'].mean().sort_values(ascending=False)
                fig_channel = px.bar(
                    x=clv_by_channel.index,
                    y=clv_by_channel.values,
                    title="Average CLV by Acquisition Channel",
                    labels={'x': 'Acquisition Channel', 'y': 'Average CLV Score'},
                    color=clv_by_channel.values,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_channel, use_container_width=True)
            
            # Feature Importance (simulated)
            st.subheader("🔍 What Drives Customer Value?")
            
            feature_importance = pd.DataFrame({
                'feature': ['Monetary Value', 'Purchase Frequency', 'Recency', 'Country', 'Age Group'],
                'importance': [0.35, 0.28, 0.15, 0.12, 0.10]
            }).sort_values('importance', ascending=False)
            
            fig_importance = px.bar(
                feature_importance,
                x='importance',
                y='feature',
                orientation='h',
                title="Feature Importance for CLV Prediction",
                color='importance',
                color_continuous_scale='Plasma'
            )
            st.plotly_chart(fig_importance, use_container_width=True)
    
    def show_retention_analysis(self):
        """Display customer retention insights"""
        st.header("📊 Customer Retention Analysis")
        
        # Calculate basic retention metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            churn_rate = (self.customer_data['recency'] > 90).mean() * 100
            st.metric("90-Day Churn Rate", f"{churn_rate:.1f}%")
        
        with col2:
            active_customers = (self.customer_data['recency'] <= 30).sum()
            st.metric("Active Customers (30d)", f"{active_customers:,}")
        
        with col3:
            avg_customer_lifetime = self.customer_data['frequency'].sum() / len(self.customer_data)
            st.metric("Avg Customer Lifetime", f"{avg_customer_lifetime:.1f} purchases")
        
        # Retention by segment
        st.subheader("🔄 Retention by Customer Segment")
        
        retention_by_segment = self.customer_data.groupby('rfm_segment').agg({
            'recency': 'mean',
            'frequency': 'mean'
        }).round(1)
        
        fig_retention = go.Figure(data=[
            go.Bar(name='Avg Recency (days)', x=retention_by_segment.index, y=retention_by_segment['recency']),
            go.Bar(name='Avg Frequency', x=retention_by_segment.index, y=retention_by_segment['frequency'])
        ])
        
        fig_retention.update_layout(
            title="Retention Metrics by RFM Segment",
            barmode='group'
        )
        st.plotly_chart(fig_retention, use_container_width=True)
    
    def show_geographic_analysis(self):
        """Display geographic distribution of customers"""
        st.header("🌍 Geographic Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            country_dist = self.customer_data['country'].value_counts()
            fig_country = px.pie(
                values=country_dist.values,
                names=country_dist.index,
                title="Customer Distribution by Country",
                color_discrete_sequence=px.colors.sequential.Plasma
            )
            st.plotly_chart(fig_country, use_container_width=True)
        
        with col2:
            clv_by_country = self.customer_data.groupby('country')['monetary'].mean().sort_values(ascending=False)
            fig_clv_country = px.bar(
                x=clv_by_country.index,
                y=clv_by_country.values,
                title="Average Customer Value by Country",
                labels={'x': 'Country', 'y': 'Average Spending ($)'},
                color=clv_by_country.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_clv_country, use_container_width=True)
    
    def show_executive_summary(self):
        """Display executive summary"""
        st.header("📋 Executive Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Key Insights")
            
            # Top performing segment
            top_segment = self.customer_data.groupby('rfm_segment')['monetary'].mean().idxmax()
            top_segment_value = self.customer_data.groupby('rfm_segment')['monetary'].mean().max()
            
            st.info(f"**🏆 Best Performing Segment:** {top_segment} (${top_segment_value:,.0f} avg value)")
            
            # Most valuable acquisition channel
            best_channel = self.customer_data.groupby('acquisition_channel')['monetary'].mean().idxmax()
            best_channel_value = self.customer_data.groupby('acquisition_channel')['monetary'].mean().max()
            
            st.success(f"**📈 Most Valuable Channel:** {best_channel} (${best_channel_value:,.0f} avg value)")
            
            # Risk area
            at_risk_count = (self.customer_data['rfm_segment'] == 'At Risk').sum()
            st.warning(f"**⚠️ Customers at Risk:** {at_risk_count} customers need attention")
        
        with col2:
            st.subheader("💡 Recommended Actions")
            
            recommendations = [
                "🎯 Focus marketing budget on high-CLV acquisition channels",
                "📊 Develop personalized campaigns for each customer segment", 
                "🔄 Implement win-back strategy for at-risk customers",
                "🏆 Create VIP program for champion customers",
                "📈 Optimize product recommendations based on segment behavior"
            ]
            
            for rec in recommendations:
                st.write(f"• {rec}")
    
    def run(self):
        """Run the complete dashboard"""
        # Show header
        self.show_header()
        
        # Setup sidebar
        filters = self.setup_sidebar()
        
        # Create navigation
        page = st.sidebar.selectbox(
            "Navigate to:",
            ["📊 Dashboard Overview", "👥 Customer Segments", "💰 CLV Analysis", 
             "📊 Retention Analysis", "🌍 Geographic View", "📋 Executive Summary"]
        )
        
        # Display selected page
        if page == "📊 Dashboard Overview":
            self.show_kpi_metrics()
            self.show_customer_segmentation()
            self.show_clv_analysis()
            
        elif page == "👥 Customer Segments":
            self.show_customer_segmentation()
            
        elif page == "💰 CLV Analysis":
            self.show_clv_analysis()
            
        elif page == "📊 Retention Analysis":
            self.show_retention_analysis()
            
        elif page == "🌍 Geographic View":
            self.show_geographic_analysis()
            
        elif page == "📋 Executive Summary":
            self.show_executive_summary()
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.info(
            "**💡 Demo Note:** This dashboard shows sample data. "
            "In production, it would connect to your live customer database."
        )

# Run the dashboard
if __name__ == "__main__":
    dashboard = StreamlitCustomerAnalytics()
    dashboard.run()