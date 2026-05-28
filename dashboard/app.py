import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

# Set page configuration with a premium look
st.set_page_config(
    page_title="EcoCast: Real-Time Air Quality & PM2.5 Prediction Dashboard",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Glassmorphism and Styling CSS
st.markdown("""
<style>
    /* Premium font and backgrounds */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    /* Header styling */
    .title-container {
        padding: 2rem;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        border: 1px rgba(255, 255, 255, 0.1) solid;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(10px);
        margin-bottom: 2rem;
        text-align: center;
        background-image: linear-gradient(to right, #4338ca, #3b82f6, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Premium Metric Box */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.15);
        transition: transform 0.3s ease;
        margin-bottom: 1rem;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .lr-card { border-left-color: #3b82f6; }
    .rf-card { border-left-color: #10b981; }
    .xgb-card { border-left-color: #f59e0b; }
    
    .metric-title {
        font-size: 0.9rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
    }
    
    .metric-unit {
        font-size: 1rem;
        font-weight: 400;
        color: #cbd5e1;
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 10rem;
        margin-top: 0.5rem;
    }
    .bg-good { background-color: #10b981; color: white; }
    .bg-moderate { background-color: #eab308; color: black; }
    .bg-poor { background-color: #f97316; color: white; }
    .bg-hazardous { background-color: #ef4444; color: white; }
</style>
""", unsafe_allow_html=True)

# Helper function to load models and scaler
@st.cache_resource
def load_ml_assets():
    try:
        scaler = joblib.load("models/scaler.joblib")
        lr_model = joblib.load("models/linear_regression.joblib")
        rf_model = joblib.load("models/random_forest.joblib")
        xgb_model = joblib.load("models/xgboost.joblib")
        comparison_df = pd.read_csv("models/model_comparison.csv")
        return scaler, lr_model, rf_model, xgb_model, comparison_df
    except Exception as e:
        st.error(f"Error loading models or scaler: {e}")
        st.warning("Please run 'python src/train.py' first to train and export the models.")
        return None, None, None, None, None

# Load the assets
scaler, lr, rf, xgb, comparison = load_ml_assets()

# --- Title Header ---
st.markdown('<div class="title-container"><h1>🍃 EcoCast: Advanced Air Quality Prediction</h1><p style="color: #94a3b8; font-size: 1.1rem; margin-top: -0.5rem;">COM2502 Course Project Demo — Predict Real-Time PM2.5 Concentrations</p></div>', unsafe_allow_html=True)

if scaler is not None:
    # --- Sidebar for User Inputs ---
    st.sidebar.markdown("### 🎛️ Simulation Parameters")
    st.sidebar.write("Adjust the meteorology and urban factors below to perform a **Live Single-Sample Regression Prediction**:")
    
    # Numerical Inputs
    temp = st.sidebar.slider("Temperature (°C)", min_value=-10.0, max_value=45.0, value=20.0, step=0.5)
    feels_like = st.sidebar.slider("Feels-Like Temp (°C)", min_value=-15.0, max_value=50.0, value=20.0, step=0.5)
    pressure = st.sidebar.slider("Atmospheric Pressure (hPa)", min_value=950, max_value=1050, value=1013, step=1)
    humidity = st.sidebar.slider("Humidity (%)", min_value=0, max_value=100, value=65, step=1)
    wind_speed = st.sidebar.slider("Wind Speed (m/s)", min_value=0.0, max_value=25.0, value=3.5, step=0.1)
    clouds_all = st.sidebar.slider("Cloud Cover (%)", min_value=0, max_value=100, value=40, step=1)
    visibility = st.sidebar.slider("Visibility (meters)", min_value=0, max_value=10000, value=10000, step=100)
    
    # City population as traffic density proxy
    city_population = st.sidebar.number_input("City Population (Traffic/Size Proxy)", min_value=1000, max_value=30000000, value=1000000, step=50000)
    
    # Categorical / Temporal Inputs
    month = st.sidebar.selectbox("Month", options=list(range(1, 13)), index=4, format_func=lambda m: ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"][m-1])
    hour = st.sidebar.slider("Hour of Day", min_value=0, max_value=23, value=12, step=1)
    day_of_week = st.sidebar.selectbox("Day of Week", options=list(range(0, 7)), index=2, format_func=lambda d: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][d])
    
    # Checkbox for weekend
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # --- Prediction Panel (Main Layout) ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🔮 Real-Time Predictions (Single Sample Demo)")
        st.write("This section demonstrates the core regression model capability. The inputs in the sidebar are scaled in real time, and all three models make predictions concurrently:")
        
        # Build features dataframe matching training feature order
        features = pd.DataFrame([{
            "temp": temp,
            "feels_like": feels_like,
            "pressure": pressure,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "clouds_all": clouds_all,
            "visibility": visibility,
            "city_population": city_population,
            "is_weekend": is_weekend,
            "month": month,
            "hour": hour,
            "day_of_week": day_of_week
        }])
        
        # Scale input features
        features_scaled = scaler.transform(features)
        
        # Predict
        pred_lr = lr.predict(features_scaled)[0]
        pred_rf = rf.predict(features_scaled)[0]
        pred_xgb = xgb.predict(features_scaled)[0]
        
        # Clip negative predictions to 0 for physical feasibility
        pred_lr = max(0, pred_lr)
        pred_rf = max(0, pred_rf)
        pred_xgb = max(0, pred_xgb)
        
        # Define alert color thresholds for PM2.5 (Standard WHO/EPA scales)
        def get_pm25_badge(pm_val):
            if pm_val <= 12:
                return '<span class="badge bg-good">GOOD</span>', "Healthy air quality. Low risk."
            elif pm_val <= 35.4:
                return '<span class="badge bg-moderate">MODERATE</span>', "Acceptable air quality. Moderate risk."
            elif pm_val <= 55.4:
                return '<span class="badge bg-poor">POOR</span>', "Unhealthy for sensitive groups."
            else:
                return '<span class="badge bg-hazardous">HAZARDOUS</span>', "Very unhealthy. High pollution level!"

        # Show three model prediction metric cards
        # 1. Linear Regression
        lr_badge, lr_text = get_pm25_badge(pred_lr)
        st.markdown(f"""
        <div class="metric-card lr-card">
            <div class="metric-title">Linear Regression (Baseline)</div>
            <div class="metric-value">{pred_lr:.2f} <span class="metric-unit">µg/m³</span></div>
            <div>{lr_badge} <span style="font-size: 0.9rem; color: #cbd5e1; margin-left: 10px;">{lr_text}</span></div>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. Random Forest
        rf_badge, rf_text = get_pm25_badge(pred_rf)
        st.markdown(f"""
        <div class="metric-card rf-card">
            <div class="metric-title">Random Forest Regressor (Ensemble)</div>
            <div class="metric-value">{pred_rf:.2f} <span class="metric-unit">µg/m³</span></div>
            <div>{rf_badge} <span style="font-size: 0.9rem; color: #cbd5e1; margin-left: 10px;">{rf_text}</span></div>
        </div>
        """, unsafe_allow_html=True)
        
        # 3. XGBoost
        xgb_badge, xgb_text = get_pm25_badge(pred_xgb)
        st.markdown(f"""
        <div class="metric-card xgb-card">
            <div class="metric-title">XGBoost Regressor (Gradient Boosting)</div>
            <div class="metric-value">{pred_xgb:.2f} <span class="metric-unit">µg/m³</span></div>
            <div>{xgb_badge} <span style="font-size: 0.9rem; color: #cbd5e1; margin-left: 10px;">{xgb_text}</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.subheader("📊 Model Performance Evaluation")
        st.write("Below are the model comparisons evaluated on the 20% holdout testing split. This validates your role as the **Model Architect**:")
        
        # Show comparison table
        st.dataframe(comparison.style.highlight_max(subset=['R2'], color='#065f46')
                                       .highlight_min(subset=['MAE', 'RMSE'], color='#065f46'))
        
        # Visualizing R2 and RMSE in charts
        st.markdown("##### Performance Metrics Comparison")
        metrics_df = comparison.melt(id_vars="Model", value_vars=["MAE", "RMSE", "R2"], var_name="Metric", value_name="Value")
        
        # Display small chart for R2
        r2_only = comparison[["Model", "R2"]]
        st.bar_chart(data=r2_only.set_index("Model"), height=180)
        st.caption("Figure 1: R-squared ($R^2$) Score by Model (Higher is better, 1.0 is perfect)")
        
    # --- Tabbed Section for Detailed Plots ---
    st.markdown("---")
    st.subheader("📈 Deep Dive Analysis & Model Diagnostics")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Actual vs. Predicted", "🔍 Feature Importance", "📉 Residuals Analysis"])
    
    # Helper to check and show plot
    def show_plot(model_name, plot_type):
        filename = f"plots/{model_name.lower().replace(' ', '_')}_{plot_type}.png"
        if os.path.exists(filename):
            try:
                img = Image.open(filename)
                st.image(img, caption=f"{model_name}: {plot_type.replace('_', ' ').title()}", use_container_width=True)
            except Exception as e:
                st.error(f"Error loading plot image: {e}")
        else:
            st.warning(f"Diagnostic plot not found at {filename}. Run train.py first.")

    with tab1:
        st.write("The scatter plot of actual vs. predicted values helps us see how closely our predictions track the y=x (perfect prediction) red line.")
        c1, c2, c3 = st.columns(3)
        with c1: show_plot("Linear Regression", "actual_vs_predicted")
        with c2: show_plot("Random Forest", "actual_vs_predicted")
        with c3: show_plot("XGBoost", "actual_vs_predicted")
        
    with tab2:
        st.write("Feature Importance charts reveal which parameters have the greatest predictive power for PM2.5 levels. XGBoost and Random Forest utilize decision split purity, while Linear Regression displays normalized coefficients.")
        c1, c2, c3 = st.columns(3)
        with c1: show_plot("Linear Regression", "feature_importance")
        with c2: show_plot("Random Forest", "feature_importance")
        with c3: show_plot("XGBoost", "feature_importance")
        
    with tab3:
        st.write("Residual plots map the error values against the predictions. A good model should have residual points randomly dispersed around the horizontal zero line, indicating homoscedasticity.")
        c1, c2, c3 = st.columns(3)
        with c1: show_plot("Linear Regression", "residuals")
        with c2: show_plot("Random Forest", "residuals")
        with c3: show_plot("XGBoost", "residuals")
        
    # Footer and Course Metadata
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.9rem; padding: 1rem;">
        Developed for **COM2502: Introduction to Data Science** Course Project Assignment.<br>
        Group Members: Person 1 (Data Engineer), **Person 2 (Model Architect)**, Person 3 (Frontend & Viz)<br>
        Submission Deadline: 14/06/2026 | Presentation: 01/06/2026
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("Models are training or not yet saved. Please run `train.py` first to generate models and diagnostic plots!")
