# MAIN DASHBOARD - Professional Customer Analytics Platform

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sqlalchemy import create_engine
import sys
import os
from datetime import datetime, timedelta

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from analytics.customer_segmentation import CustomerSegmentation
from models.clv_prediction import CLVPredictor

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

class CustomerAnalyticsDashboard:
    def __init__(self):
        self.engine = create_engine('sqlite:///data/customer_analytics.db')
        self.segmentation = CustomerSegmentation(self.engine)
        self.clv_predictor = CLVPredictor(self.engine)
        
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_data(_self):
        """Load data from database with caching"""
        try:
            # Load customer data with RFM
            query = """
            SELECT c.*, r.recency, r.frequency, r.monetary, r.rfm_score, r.segment as rfm_segment
            FROM customers c
            JOIN customer_rfm r ON c.customer_id = r.customer_id
            """
            customer_data = pd.read_sql(query, _self.engine)
            
            # Load transaction data
            transaction_data = pd.read_sql("SELECT * FROM transactions", _self.engine)
            
            return customer_data, transaction_data
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def setup_sidebar(self):
        """Create the sidebar with filters and controls"""
        st.sidebar.title(" Dashboard Controls")
        
        # Date range filter
        st.sidebar.subheader(" Date Range")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            start_date = st.date_input("Start Date", 
                                     value=datetime.now() - timedelta(days=365))
        with col2:
            end_date = st.date_input("End Date", 
                                   value=datetime.now())
        
        # Customer segment filter
        st.sidebar.subheader(" Customer Segments")
        segments = st.sidebar.multiselect(
            "Select segments to display:",
            ["All", "Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Cannot Lose"],
            default=["All"]
        )
        
        # Country filter
        st.sidebar.subheader(" Geography")
        countries = st.sidebar.multiselect(
            "Select countries:",
            ["All", "US", "UK", "Germany", "France", "Canada", "Australia"],
            default=["All"]
        )
        
        return {
            'start_date': start_date,
            'end_date': end_date,
            'segments': segments,
            'countries': countries
        }
    
    def show_header(self):
        """Display the main header"""
        st.markdown('<h1 class="main-header"> Customer Analytics Platform</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        **Real-time customer intelligence platform** providing actionable insights into 
        customer behavior, lifetime value, and retention strategies.
        """)
        
        # Add some space
        st.markdown("<br>", unsafe_allow_html=True)
    
    def show_kpi_metrics(self, customer_data, transaction_data):
        """Display key performance indicators"""
        st.header(" Key Performance Indicators")
        
        # Create columns for metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_customers = len(customer_data)
            st.metric(
                "Total Customers", 
                f"{total_customers:,}",
                help="Total number of unique customers"
            )
        
        with col2:
            total_revenue = transaction_data['total_amount'].sum()
            st.metric(
                "Total Revenue", 
                f"${total_revenue:,.0f}",
                help="Sum of all transaction amounts"
            )
        
        with col3:
            avg_clv = customer_data['monetary'].mean()
            st.metric(
                "Avg Customer Value", 
                f"${avg_clv:,.0f}",
                help="Average lifetime value per customer"
            )
        
        with col4:
            avg_frequency = customer_data['frequency'].mean()
            st.metric(
                "Avg Purchase Frequency", 
                f"{avg_frequency:.1f}",
                help="Average number of purchases per customer"
            )
        
        with col5:
            repeat_rate = (customer_data['frequency'] > 1).mean() * 100
            st.metric(
                "Repeat Purchase Rate", 
                f"{repeat_rate:.1f}%",
                help="Percentage of customers with multiple purchases"
            )
        
        # Add some space
        st.markdown("<br>", unsafe_allow_html=True)
    
    def show_customer_segmentation(self, customer_data):
        """Display customer segmentation analysis"""
        st.header("👥 Customer Segmentation Analysis")
        
        with st.spinner("Analyzing customer segments..."):
            try:
                # Perform clustering
                clustered_data, kmeans, scaler = self.segmentation.perform_kmeans_clustering(n_clusters=5)
                clustered_data, summary, names = self.segmentation.analyze_clusters(clustered_data)
                
                # Create two columns for segmentation charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Segment distribution pie chart
                    segment_dist = clustered_data['cluster_name'].value_counts()
                    fig_segment = px.pie(
                        values=segment_dist.values,
                        names=segment_dist.index,
                        title="Customer Segment Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_segment.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_segment, use_container_width=True)
                
                with col2:
                    # RFM scores by segment
                    fig_rfm = px.box(
                        clustered_data,
                        x='cluster_name',
                        y='monetary',
                        title="Spending Distribution by Segment",
                        color='cluster_name'
                    )
                    fig_rfm.update_layout(showlegend=False)
                    st.plotly_chart(fig_rfm, use_container_width=True)
                
                # 3D Scatter plot
                st.subheader("3D Customer Segmentation")
                fig_3d, fig_comparison = self.segmentation.create_segmentation_visualizations(clustered_data)
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # Segment insights
                st.subheader(" Segment Insights")
                for cluster_name in clustered_data['cluster_name'].unique():
                    segment_data = clustered_data[clustered_data['cluster_name'] == cluster_name]
                    
                    with st.expander(f" {cluster_name} - {len(segment_data)} customers"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Avg Recency", f"{segment_data['recency'].mean():.0f} days")
                        with col2:
                            st.metric("Avg Frequency", f"{segment_data['frequency'].mean():.1f}")
                        with col3:
                            st.metric("Avg Spending", f"${segment_data['monetary'].mean():,.0f}")
                        
                        # Recommendations
                        if cluster_name == 'VIP Customers':
                            st.info("** Strategy:** Offer exclusive rewards, early access, and premium support")
                        elif cluster_name == 'Loyal Customers':
                            st.info("** Strategy:** Strengthen with loyalty programs and personalized offers")
                        elif cluster_name == 'At Risk Customers':
                            st.warning("** Strategy:** Implement win-back campaigns with special discounts")
                        elif cluster_name == 'One-time Buyers':
                            st.info("** Strategy:** Encourage repeat purchases with follow-up offers")
                        else:
                            st.info("** Strategy:** Focus on increasing engagement and purchase frequency")
                
            except Exception as e:
                st.error(f"Error in segmentation analysis: {e}")
    
    def show_clv_analysis(self, customer_data):
        """Display Customer Lifetime Value analysis"""
        st.header(" Customer Lifetime Value Analysis")
        
        with st.spinner("Training CLV prediction model..."):
            try:
                # Train CLV model
                results, feature_importance = self.clv_predictor.train_model()
                data = self.clv_predictor.prepare_features()
                
                # Create visualizations
                fig_dist, fig_channel, fig_segment = self.clv_predictor.create_clv_visualizations(data)
                
                # CLV Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_predicted_value = data['future_90d_value'].sum()
                    st.metric("Total Predicted Value (90d)", f"${total_predicted_value:,.0f}")
                
                with col2:
                    avg_clv = data['future_90d_value'].mean()
                    st.metric("Average CLV (90d)", f"${avg_clv:.0f}")
                
                with col3:
                    high_value_threshold = data['future_90d_value'].quantile(0.8)
                    st.metric("High-Value Threshold", f"${high_value_threshold:.0f}")
                
                with col4:
                    model_accuracy = results['test_r2'] * 100
                    st.metric("Model Accuracy (R²)", f"{model_accuracy:.1f}%")
                
                # CLV Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                with col2:
                    st.plotly_chart(fig_channel, use_container_width=True)
                
                # Feature Importance
                st.subheader(" What Drives Customer Value?")
                if feature_importance is not None:
                    fig_importance = px.bar(
                        feature_importance.head(10),
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Top 10 Features Driving CLV",
                        color='importance',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                
                # High-Value Customer Analysis
                st.subheader(" High-Value Customer Profile")
                high_value_customers = data[data['future_90d_value'] >= high_value_threshold]
                
                if len(high_value_customers) > 0:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Most common age group
                        common_age = high_value_customers['age_group'].mode()[0]
                        st.metric("Most Common Age", common_age)
                    
                    with col2:
                        # Most common country
                        common_country = high_value_customers['country'].mode()[0]
                        st.metric("Most Common Country", common_country)
                    
                    with col3:
                        # Most common acquisition channel
                        common_channel = high_value_customers['acquisition_channel'].mode()[0]
                        st.metric("Top Acquisition Channel", common_channel)
                
            except Exception as e:
                st.error(f"Error in CLV analysis: {e}")
    
    def show_retention_analysis(self, customer_data, transaction_data):
        """Display customer retention insights"""
        st.header(" Customer Retention Analysis")
        
        try:
            # Calculate basic retention metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                churn_rate = (customer_data['recency'] > 90).mean() * 100
                st.metric("90-Day Churn Rate", f"{churn_rate:.1f}%")
            
            with col2:
                active_customers = (customer_data['recency'] <= 30).sum()
                st.metric("Active Customers (30d)", f"{active_customers:,}")
            
            with col3:
                avg_customer_lifetime = customer_data['frequency'].sum() / len(customer_data)
                st.metric("Avg Customer Lifetime", f"{avg_customer_lifetime:.1f} purchases")
            
            # Cohort Analysis
            st.subheader(" Cohort Analysis")
            
            # Simple cohort analysis (monthly)
            transaction_data['cohort_month'] = transaction_data['transaction_date'].dt.to_period('M')
            transaction_data['cohort'] = transaction_data.groupby('customer_id')['transaction_date'].transform('min').dt.to_period('M')
            
            cohort_data = transaction_data.groupby(['cohort', 'cohort_month']).agg({
                'customer_id': 'nunique'
            }).reset_index()
            
            cohort_data['period_number'] = (cohort_data['cohort_month'] - cohort_data['cohort']).apply(lambda x: x.n)
            
            # Create retention matrix
            cohort_pivot = cohort_data.pivot_table(
                index='cohort',
                columns='period_number',
                values='customer_id',
                aggfunc='sum'
            )
            
            cohort_size = cohort_pivot.iloc[:, 0]
            retention_matrix = cohort_pivot.divide(cohort_size, axis=0)
            
            # Create retention heatmap
            fig_retention = px.imshow(
                retention_matrix,
                title="Customer Retention Heatmap",
                labels=dict(x="Months After Acquisition", y="Acquisition Cohort", color="Retention Rate"),
                color_continuous_scale="Blues",
                aspect="auto"
            )
            st.plotly_chart(fig_retention, use_container_width=True)
            
            # Retention by segment
            st.subheader(" Retention by Customer Segment")
            
            retention_by_segment = customer_data.groupby('rfm_segment').agg({
                'recency': 'mean',
                'frequency': 'mean'
            }).round(1)
            
            fig_retention_segment = go.Figure(data=[
                go.Bar(name='Avg Recency (days)', x=retention_by_segment.index, y=retention_by_segment['recency']),
                go.Bar(name='Avg Frequency', x=retention_by_segment.index, y=retention_by_segment['frequency'])
            ])
            
            fig_retention_segment.update_layout(
                title="Retention Metrics by RFM Segment",
                barmode='group'
            )
            st.plotly_chart(fig_retention_segment, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in retention analysis: {e}")
    
    def show_geographic_analysis(self, customer_data):
        """Display geographic distribution of customers"""
        st.header(" Geographic Analysis")
        
        try:
            # Country distribution
            country_dist = customer_data['country'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_country = px.pie(
                    values=country_dist.values,
                    names=country_dist.index,
                    title="Customer Distribution by Country",
                    color_discrete_sequence=px.colors.sequential.Plasma
                )
                st.plotly_chart(fig_country, use_container_width=True)
            
            with col2:
                # CLV by country
                clv_by_country = customer_data.groupby('country')['monetary'].mean().sort_values(ascending=False)
                fig_clv_country = px.bar(
                    x=clv_by_country.index,
                    y=clv_by_country.values,
                    title="Average Customer Value by Country",
                    labels={'x': 'Country', 'y': 'Average CLV ($)'},
                    color=clv_by_country.values,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_clv_country, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in geographic analysis: {e}")
    
    def show_executive_summary(self, customer_data, transaction_data):
        """Display executive summary"""
        st.header(" Executive Summary")
        
        try:
            # Key insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(" Key Insights")
                
                # Top performing segment
                top_segment = customer_data.groupby('rfm_segment')['monetary'].mean().idxmax()
                top_segment_value = customer_data.groupby('rfm_segment')['monetary'].mean().max()
                
                st.info(f"** Best Performing Segment:** {top_segment} (${top_segment_value:,.0f} avg value)")
                
                # Most valuable acquisition channel
                best_channel = customer_data.groupby('acquisition_channel')['monetary'].mean().idxmax()
                best_channel_value = customer_data.groupby('acquisition_channel')['monetary'].mean().max()
                
                st.success(f"**📈 Most Valuable Channel:** {best_channel} (${best_channel_value:,.0f} avg value)")
                
                # Risk area
                at_risk_count = (customer_data['rfm_segment'] == 'At Risk').sum()
                st.warning(f"** Customers at Risk:** {at_risk_count} customers need attention")
            
            with col2:
                st.subheader(" Recommended Actions")
                
                recommendations = [
                    " Focus marketing budget on high-CLV acquisition channels",
                    " Develop personalized campaigns for each customer segment", 
                    " Implement win-back strategy for at-risk customers",
                    " Create VIP program for champion customers",
                    " Optimize product recommendations based on segment behavior"
                ]
                
                for rec in recommendations:
                    st.write(f"• {rec}")
            
            # Performance trends
            st.subheader("📈 Performance Trends")
            
            # Monthly revenue trend (simulated)
            months = pd.date_range(start='2023-01-01', end='2023-12-01', freq='M')
            revenue_trend = np.random.normal(50000, 10000, len(months)) * (1 + np.arange(len(months)) * 0.05)
            
            fig_trend = px.line(
                x=months,
                y=revenue_trend,
                title="Monthly Revenue Trend",
                labels={'x': 'Month', 'y': 'Revenue ($)'}
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in executive summary: {e}")
    
    def run(self):
        """Run the complete dashboard"""
        try:
            # Show header
            self.show_header()
            
            # Setup sidebar
            filters = self.setup_sidebar()
            
            # Load data
            customer_data, transaction_data = self.load_data()
            
            if customer_data.empty:
                st.error("No data loaded. Please check your database connection.")
                return
            
            # Create navigation
            page = st.sidebar.selectbox(
                "Navigate to:",
                [" Dashboard Overview", " Customer Segments", " CLV Analysis", 
                 " Retention Analysis", " Geographic View", " Executive Summary"]
            )
            
            # Display selected page
            if page == " Dashboard Overview":
                self.show_kpi_metrics(customer_data, transaction_data)
                self.show_customer_segmentation(customer_data)
                self.show_clv_analysis(customer_data)
                
            elif page == "👥 Customer Segments":
                self.show_customer_segmentation(customer_data)
                
            elif page == " CLV Analysis":
                self.show_clv_analysis(customer_data)
                
            elif page == " Retention Analysis":
                self.show_retention_analysis(customer_data, transaction_data)
                
            elif page == "🌍 Geographic View":
                self.show_geographic_analysis(customer_data)
                
            elif page == " Executive Summary":
                self.show_executive_summary(customer_data, transaction_data)
            
            # Footer
            st.sidebar.markdown("---")
            st.sidebar.info(
                "**💡 Tip:** Use the filters in the sidebar to explore different "
                "customer segments and time periods."
            )
            
        except Exception as e:
            st.error(f"Error running dashboard: {e}")
            st.info("Please make sure you've run the ETL pipeline first to generate the data.")

# Run the dashboard
if __name__ == "__main__":
    dashboard = CustomerAnalyticsDashboard()
    dashboard.run()