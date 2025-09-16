# streamlit_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# Load data and models
@st.cache_data
def load_data():
    try:
        results_df = pd.read_csv('model_predictions_results.csv')
        results_df['Date'] = pd.to_datetime(results_df['Date'])
        
        processed_df = pd.read_csv('dahi_plant_processed.csv')
        processed_df['Date'] = pd.to_datetime(processed_df['Date'])
        
        model = joblib.load('high_waste_predictor_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        
        return results_df, processed_df, model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

# Main dashboard
def main():
    # Header
    st.markdown('<h1 class="main-header">🏭 Nestlé Dahi Plant Waste Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    results_df, processed_df, model, scaler, feature_names = load_data()
    
    if results_df is None:
        st.error("Please run Phases 1-3 first to generate the required data files.")
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
    
    # Main content
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
            cost_savings = (high_waste_avg - filtered_df[filtered_df['Is_High_Waste'] == 0]['Total_Waste_kg'].mean()) * 100 * high_waste_days
            st.metric("Potential Savings", f"₹{cost_savings:,.0f}")
        
        # Waste trend chart
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
        
        # Highlight prediction errors
        false_negatives = filtered_df[(filtered_df['Is_High_Waste'] == 1) & (filtered_df['Predicted_High_Waste'] == 0)]
        false_positives = filtered_df[(filtered_df['Is_High_Waste'] == 0) & (filtered_df['Predicted_High_Waste'] == 1)]
        
        if len(false_negatives) > 0:
            fig.add_trace(go.Scatter(
                x=false_negatives['Date'],
                y=false_negatives['Total_Waste_kg'],
                mode='markers',
                name='False Negatives',
                marker=dict(color='orange', size=10, symbol='x')
            ))
        
        if len(false_positives) > 0:
            fig.add_trace(go.Scatter(
                x=false_positives['Date'],
                y=false_positives['Total_Waste_kg'],
                mode='markers',
                name='False Positives',
                marker=dict(color='purple', size=10, symbol='circle')
            ))
        
        fig.update_layout(
            title='Waste Trends with Prediction Performance',
            xaxis_title='Date',
            yaxis_title='Total Waste (kg)',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        if model and feature_names:
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': model.coef_[0],
                'Impact': np.abs(model.coef_[0])
            }).sort_values('Impact', ascending=True)
            
            fig = go.Figure(go.Bar(
                y=feature_importance['Feature'],
                x=feature_importance['Coefficient'],
                orientation='h',
                marker_color=['green' if x > 0 else 'red' for x in feature_importance['Coefficient']]
            ))
            
            fig.update_layout(
                title='Key Risk Factors (Positive = Increases Risk)',
                xaxis_title='Impact on High-Waste Probability',
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Prediction Details")
        
        # Prediction performance
        col1, col2 = st.columns(2)
        
        with col1:
            # Confusion matrix
            cm = np.zeros((2, 2))
            if len(filtered_df) > 0:
                tn = len(filtered_df[(filtered_df['Is_High_Waste'] == 0) & (filtered_df['Predicted_High_Waste'] == 0)])
                fp = len(filtered_df[(filtered_df['Is_High_Waste'] == 0) & (filtered_df['Predicted_High_Waste'] == 1)])
                fn = len(filtered_df[(filtered_df['Is_High_Waste'] == 1) & (filtered_df['Predicted_High_Waste'] == 0)])
                tp = len(filtered_df[(filtered_df['Is_High_Waste'] == 1) & (filtered_df['Predicted_High_Waste'] == 1)])
                
                fig = go.Figure(data=go.Heatmap(
                    z=[[tn, fp], [fn, tp]],
                    x=['Predicted Normal', 'Predicted High'],
                    y=['Actual Normal', 'Actual High'],
                    colorscale='Blues',
                    text=[[f"TN: {tn}", f"FP: {fp}"], [f"FN: {fn}", f"TP: {tp}"]],
                    texttemplate="%{text}",
                    textfont={"size": 16}
                ))
                
                fig.update_layout(
                    title='Confusion Matrix',
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Accuracy by day of week
            if 'Day_of_Week' not in filtered_df.columns:
                filtered_df['Day_of_Week'] = filtered_df['Date'].dt.day_name()
            
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
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed predictions table
        st.subheader("Detailed Predictions")
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
        
        col1, col2 = st.columns(2)
        
        with col1:
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
        
        with col2:
            # Prediction confidence
            fig = px.histogram(
                filtered_df,
                x='Prediction_Probability',
                color='Correct_Prediction',
                nbins=20,
                title='Prediction Confidence Distribution',
                labels={'Prediction_Probability': 'Prediction Probability'},
                color_discrete_map={1: 'green', 0: 'red'}
            )
            fig.add_vline(x=0.5, line_dash="dash", line_color="black")
            st.plotly_chart(fig, use_container_width=True)
        
        # Time series analysis
        st.subheader("Time Series Analysis")
        
        # Rolling average
        filtered_df['7D_Avg'] = filtered_df['Total_Waste_kg'].rolling(window=7).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['Total_Waste_kg'],
            name='Daily Waste',
            line=dict(color='lightblue', width=1)
        ))
        fig.add_trace(go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['7D_Avg'],
            name='7-Day Average',
            line=dict(color='blue', width=3)
        ))
        fig.add_hline(y=threshold, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title='Waste Trends with 7-Day Moving Average',
            xaxis_title='Date',
            yaxis_title='Waste (kg)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Actionable Recommendations")
        
        # Calculate insights
        high_waste_days = filtered_df[filtered_df['Is_High_Waste'] == 1]
        if len(high_waste_days) > 0:
            most_common_day = high_waste_days['Date'].dt.day_name().mode()[0]
        
        # Recommendations
        st.markdown("""
        <div class="insight-box">
            <h3>🎯 Key Insights</h3>
            <ul>
                <li>Model accuracy: <strong>{accuracy:.1%}</strong></li>
                <li>Most problematic day: <strong>{most_common_day}</strong></li>
                <li>Potential monthly savings: <strong>₹{savings:,.0f}</strong></li>
            </ul>
        </div>
        """.format(
            accuracy=accuracy,
            most_common_day=most_common_day if len(high_waste_days) > 0 else "N/A",
            savings=cost_savings
        ), unsafe_allow_html=True)
        
        # Top recommendations
        st.subheader("Top Recommendations")
        
        st.markdown("""
        <div class="recommendation">
            <h4>1. Focus on {top_feature}</h4>
            <p>This is your biggest risk factor for high waste days. Implement additional monitoring and quality checks.</p>
        </div>
        """.format(top_feature=feature_names[0] if feature_names else "Key Factors"), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="recommendation">
            <h4>2. Enhance {worst_day} Operations</h4>
            <p>Prediction accuracy is lowest on this day. Consider additional staffing or process reviews.</p>
        </div>
        """.format(worst_day=accuracy_by_day.idxmin() if len(accuracy_by_day) > 0 else "Critical Days"), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="recommendation">
            <h4>3. Implement Predictive Monitoring</h4>
            <p>Use the model's daily predictions to allocate resources more effectively on high-risk days.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Action plan
        st.subheader("30-Day Action Plan")
        
        action_plan = [
            {"Week": "Week 1", "Task": "Train team on interpreting predictions", "Owner": "Plant Manager"},
            {"Week": "Week 2", "Task": "Implement enhanced checks on high-risk days", "Owner": "Quality Team"},
            {"Week": "Week 3", "Task": "Review processes for top risk factors", "Owner": "Process Engineer"},
            {"Week": "Week 4", "Task": "Measure impact and adjust strategy", "Owner": "Data Analyst"}
        ]
        
        st.table(pd.DataFrame(action_plan))
        
        # Download report
        st.download_button(
            label="📄 Download Full Report",
            data=generate_report(filtered_df, accuracy, cost_savings, feature_names),
            file_name="nestle_waste_report.md",
            mime="text/markdown"
        )

def generate_report(df, accuracy, savings, feature_names):
    report = f"""
# Nestlé Dahi Plant Waste Prediction Report

## Executive Summary
- **Model Accuracy**: {accuracy:.1%}
- **Potential Monthly Savings**: ₹{savings:,.0f}
- **High-Waste Days Identified**: {len(df[df['Is_High_Waste'] == 1])}

## Key Findings
1. Top risk factor: {feature_names[0] if feature_names else 'N/A'}
2. Average waste reduction potential: {df[df['Is_High_Waste'] == 1]['Total_Waste_kg'].mean() - df[df['Is_High_Waste'] == 0]['Total_Waste_kg'].mean():.1f} kg per high-waste day

## Recommendations
1. Implement predictive monitoring system
2. Focus resources on high-risk days
3. Review processes related to top risk factors

## Next Steps
- Expand analysis with production volume data
- Implement real-time dashboard
- Continuous model improvement

Report generated on: {datetime.now().strftime('%Y-%m-%d')}
"""
    return report

if __name__ == "__main__":
    main()
