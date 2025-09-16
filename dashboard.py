# streamlit_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Nestlé Dahi Plant - Waste Prediction Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #0066CC;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #0066CC;
        margin-bottom: 1rem;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #00A0E3;
        margin: 1rem 0;
    }
    .recommendation {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data and models with robust error handling
@st.cache_data
def load_data():
    data_loaded = False
    results_df, processed_df, model, scaler, feature_names = None, None, None, None, None
    
    try:
        # Load prediction results
        results_df = pd.read_csv('model_predictions_results.csv')
        results_df['Date'] = pd.to_datetime(results_df['Date'])
        st.success("✓ Loaded prediction results")
        
        # Load original processed data for context
        processed_df = pd.read_csv('dahi_plant_processed.csv')
        processed_df['Date'] = pd.to_datetime(processed_df['Date'])
        st.success("✓ Loaded processed data")
        
        # Try different possible filenames for model files
        model_filenames = [
            'high_waste_predictor_model.pkl',
            'high_waste_predictor_model.pld',  # Your actual filename
            'high_waste_predictor_model.plx'
        ]
        
        scaler_filenames = [
            'scaler.pkl',
            'scaler.pld',  # Your actual filename
            'scaler.plx'
        ]
        
        feature_filenames = [
            'feature_names.pkl',
            'feature_names.pld',  # Your actual filename
            'feature_names.plx'
        ]
        
        # Load model
        model = None
        for filename in model_filenames:
            try:
                model = joblib.load(filename)
                st.success(f"✓ Loaded model from {filename}")
                break
            except FileNotFoundError:
                continue
        
        # Load scaler
        scaler = None
        for filename in scaler_filenames:
            try:
                scaler = joblib.load(filename)
                st.success(f"✓ Loaded scaler from {filename}")
                break
            except FileNotFoundError:
                continue
        
        # Load feature names
        feature_names = None
        for filename in feature_filenames:
            try:
                feature_names = joblib.load(filename)
                st.success(f"✓ Loaded feature names from {filename}")
                break
            except FileNotFoundError:
                continue
        
        if model is not None and results_df is not None:
            data_loaded = True
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
    
    return results_df, processed_df, model, scaler, feature_names, data_loaded

def generate_report(df, accuracy, savings):
    high_waste_count = len(df[df['Is_High_Waste'] == 1])
    avg_high_waste = df[df['Is_High_Waste'] == 1]['Total_Waste_kg'].mean()
    avg_normal_waste = df[df['Is_High_Waste'] == 0]['Total_Waste_kg'].mean()
    
    report = f"""
# Nestlé Dahi Plant Waste Prediction Report

## Executive Summary
- **Model Accuracy**: {accuracy:.1%}
- **High-Waste Days Identified**: {high_waste_count}
- **Average High Waste**: {avg_high_waste:.1f} kg/day
- **Average Normal Waste**: {avg_normal_waste:.1f} kg/day
- **Potential Monthly Savings**: ₹{savings:,.0f}

## Key Findings
- Waste reduction potential: {avg_high_waste - avg_normal_waste:.1f} kg per high-waste day
- Overall model reliability: {'Excellent' if accuracy > 0.8 else 'Good' if accuracy > 0.7 else 'Needs improvement'}

## Actionable Recommendations

### 1. Implement Predictive Monitoring
- Use daily predictions for resource allocation
- Focus on days with >60% prediction probability
- Establish baseline metrics for comparison

### 2. Enhance High-Risk Day Procedures
- Increase quality checks by 50% on predicted high-waste days
- Review equipment calibration before high-risk periods
- Assign senior staff to monitor critical processes

### 3. Financial Impact Tracking
- Monitor actual vs predicted savings
- Set target of ₹{savings//2:,.0f} monthly savings initially
- Report savings to management weekly

## Next Steps
1. Validate model with production volume data
2. Expand to include specific waste materials
3. Implement real-time monitoring dashboard
4. Train operational team on using predictions

Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
    return report

# Main dashboard
def main():
    # Header
    st.markdown('<h1 class="main-header">🏭 Nestlé Dahi Plant Waste Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    results_df, processed_df, model, scaler, feature_names, data_loaded = load_data()
    
    if not data_loaded:
        st.error("""
        ❌ Required data files not found! 
        
        Please make sure you have these files in the same directory:
        - `model_predictions_results.csv` (from Phase 3)
        - `dahi_plant_processed.csv` (from Phase 1)
        - `high_waste_predictor_model.pld` (your model file)
        - `scaler.pld` (your scaler file) 
        - `feature_names.pld` (your feature names file)
        
        Run Phases 1-3 first to generate these files.
        """)
        
        # Show available files for debugging
        import os
        st.subheader("Available files in current directory:")
        available_files = [f for f in os.listdir('.') if os.path.isfile(f)]
        for file in available_files:
            st.write(f"• {file}")
        
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/Nestle-logo.svg/320px-Nestle-logo.svg.png", width=200)
        st.title("Dashboard Controls")
        
        # Date range selector
        min_date = results_df['Date'].min()
        max_date = results_df['Date'].max()
        selected_dates = st.date_input(
            "Select Date Range:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Filter data
        if len(selected_dates) == 2:
            start_date, end_date = selected_dates
            filtered_df = results_df[
                (results_df['Date'] >= pd.to_datetime(start_date)) & 
                (results_df['Date'] <= pd.to_datetime(end_date))
            ]
        else:
            filtered_df = results_df
        
        # Display quick stats in sidebar
        st.subheader("Quick Stats")
        total_days = len(filtered_df)
        high_waste_days = len(filtered_df[filtered_df['Is_High_Waste'] == 1])
        accuracy = filtered_df['Correct_Prediction'].mean()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Days", total_days)
        with col2:
            st.metric("High-Waste Days", high_waste_days)
        
        st.metric("Prediction Accuracy", f"{accuracy:.1%}")
    
    # Main content - FIXED THE TAB NAMES (removed # and & symbols)
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Predictions", "📈 Analytics", "🎯 Recommendations"])
    
    with tab1:
        st.header("Overview Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_waste = filtered_df['Total_Waste_kg'].mean()
            st.metric("Avg Daily Waste", f"{avg_waste:.1f} kg")
        
        with col2:
            high_waste_avg = filtered_df[filtered_df['Is_High_Waste'] == 1]['Total_Waste_kg'].mean()
            st.metric("Avg High Waste", f"{high_waste_avg:.1f} kg")
        
        with col3:
            accuracy = filtered_df['Correct_Prediction'].mean()
            st.metric("Model Accuracy", f"{accuracy:.1%}")
        
        with col4:
            normal_waste_avg = filtered_df[filtered_df['Is_High_Waste'] == 0]['Total_Waste_kg'].mean()
            cost_savings = (high_waste_avg - normal_waste_avg) * 100 * high_waste_days
            st.metric("Potential Savings", f"₹{cost_savings:,.0f}")
        
        # Waste trend chart
        import plotly.express as px
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_df['Date'], 
            y=filtered_df['Total_Waste_kg'],
            mode='lines+markers',
            name='Actual Waste',
            line=dict(color='#0066CC', width=2)
        ))
        
        # Add high-waste threshold
        threshold = filtered_df['Total_Waste_kg'].quantile(0.75)
        fig.add_hline(
            y=threshold, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"High-Waste Threshold ({threshold:.1f} kg)",
            annotation_position="bottom right"
        )
        
        fig.update_layout(
            title='Waste Trends Over Time',
            xaxis_title='Date',
            yaxis_title='Waste (kg)',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Basic insights
        st.subheader("Key Insights")
        
        if 'Day_of_Week' not in filtered_df.columns:
            filtered_df['Day_of_Week'] = filtered_df['Date'].dt.day_name()
        
        most_common_high_day = filtered_df[filtered_df['Is_High_Waste'] == 1]['Day_of_Week'].mode()
        if len(most_common_high_day) > 0:
            st.info(f"📅 **Most common high-waste day**: {most_common_high_day[0]}")
        
        st.info(f"💰 **Potential monthly savings**: ₹{cost_savings:,.0f}")
        st.info(f"🎯 **Model reliability**: {'Excellent' if accuracy > 0.8 else 'Good' if accuracy > 0.7 else 'Needs improvement'}")

    with tab2:
        st.header("Prediction Details")
        
        # Detailed predictions table
        st.subheader("Daily Predictions")
        display_df = filtered_df[['Date', 'Total_Waste_kg', 'Is_High_Waste', 'Predicted_High_Waste', 'Prediction_Probability', 'Correct_Prediction']].copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        display_df['Is_High_Waste'] = display_df['Is_High_Waste'].map({1: 'Yes', 0: 'No'})
        display_df['Predicted_High_Waste'] = display_df['Predicted_High_Waste'].map({1: 'Yes', 0: 'No'})
        display_df['Correct_Prediction'] = display_df['Correct_Prediction'].map({1: '✓', 0: '✗'})
        display_df['Prediction_Probability'] = display_df['Prediction_Probability'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(
            display_df,
            column_config={
                "Date": "Date",
                "Total_Waste_kg": st.column_config.NumberColumn("Waste (kg)", format="%.1f"),
                "Is_High_Waste": "Actual High",
                "Predicted_High_Waste": "Predicted High",
                "Prediction_Probability": "Confidence",
                "Correct_Prediction": "Correct"
            },
            hide_index=True,
            use_container_width=True
        )

    with tab3:
        st.header("Advanced Analytics")
        
        # Waste distribution
        fig = px.histogram(
            filtered_df, 
            x='Total_Waste_kg',
            nbins=20,
            title='Distribution of Daily Waste',
            labels={'Total_Waste_kg': 'Waste (kg)'}
        )
        fig.add_vline(x=threshold, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
        
        # Accuracy by day of week
        accuracy_by_day = filtered_df.groupby('Day_of_Week')['Correct_Prediction'].mean()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        accuracy_by_day = accuracy_by_day.reindex(day_order)
        
        fig = go.Figure(go.Bar(
            x=accuracy_by_day.index,
            y=accuracy_by_day.values,
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title='Prediction Accuracy by Day of Week',
            xaxis_title='Day of Week',
            yaxis_title='Accuracy',
            yaxis_tickformat='.0%',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.header("Actionable Recommendations")
        
        # Calculate actual values for recommendations
        high_waste_count = len(filtered_df[filtered_df['Is_High_Waste'] == 1])
        avg_high_waste = filtered_df[filtered_df['Is_High_Waste'] == 1]['Total_Waste_kg'].mean()
        avg_normal_waste = filtered_df[filtered_df['Is_High_Waste'] == 0]['Total_Waste_kg'].mean()
        potential_savings_per_day = (avg_high_waste - avg_normal_waste) * 100  # ₹100 per kg
        total_potential_savings = potential_savings_per_day * high_waste_count
        
        st.markdown(f"""
        <div class="recommendation">
            <h4>🎯 1. Implement Predictive Monitoring</h4>
            <p>Use the model's daily predictions to allocate resources more effectively on high-risk days. 
            The model has identified <b>{high_waste_count}</b> high-waste days with <b>{accuracy:.1%}</b> accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="recommendation">
            <h4>📊 2. Focus on High-Risk Days</h4>
            <p>Increase quality checks by 50% on predicted high-waste days. 
            High-waste days average <b>{avg_high_waste:.1f} kg</b> compared to <b>{avg_normal_waste:.1f} kg</b> on normal days.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="recommendation">
            <h4>💰 3. Track Financial Impact</h4>
            <p>Monitor savings of up to <b>₹{potential_savings_per_day:,.0f}</b> per high-waste day through predictive interventions.
            Potential monthly savings: <b>₹{total_potential_savings:,.0f}</b>.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Download report - FIXED THIS LINE
        st.download_button(
            label="📄 Download Summary Report",
            data=generate_report(filtered_df, accuracy, total_potential_savings),
            file_name="nestle_waste_report.md",
            mime="text/markdown"
        )

if __name__ == "__main__":
    main()
