import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Import visualization and metric functions
from utils import calculate_metrics, plot_actual_vs_predicted, plot_residuals, plot_feature_importance

def main():
    # Setup directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # 1. Load Dataset
    data_path = "openweather_weather_airpollution_top3cities_per_country.csv"
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 2. Define Features and Target
    # Independent variables (Predictors) - Using weather features and population as a proxy for traffic
    feature_cols = [
        "temp", "feels_like", "pressure", "humidity", "wind_speed", 
        "clouds_all", "visibility", "city_population", "is_weekend", 
        "month", "hour", "day_of_week"
    ]
    target_col = "pm2_5"
    
    print(f"Selecting features: {feature_cols}")
    print(f"Target column: {target_col}")
    
    # Drop rows where the target is missing (should be 0 anyway)
    df = df.dropna(subset=[target_col])
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # 3. Data Cleaning / Preprocessing
    # Impute missing values in visibility (with median)
    if X["visibility"].isna().sum() > 0:
        median_visibility = X["visibility"].median()
        print(f"Imputing {X['visibility'].isna().sum()} missing values in 'visibility' with median ({median_visibility})")
        X["visibility"] = X["visibility"].fillna(median_visibility)
    
    # Double check other columns
    missing_counts = X.isna().sum()
    if missing_counts.sum() > 0:
        print("Warning: Other columns contain missing values. Filling with column medians...")
        for col in X.columns:
            if X[col].isna().sum() > 0:
                X[col] = X[col].fillna(X[col].median())
                
    # 4. Train-Test Split (80-20)
    print("\nSplitting dataset into train and test sets (80-20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # 5. Feature Scaling
    print("Scaling numerical features with StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for future inference / Streamlit
    scaler_path = os.path.join("models", "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")
    
    # Initialize dictionary to collect all results
    metrics_results = []
    
    # ------------------ MODEL 1: LINEAR REGRESSION ------------------
    print("\n--- Training Model 1: Linear Regression ---")
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_lr = lr_model.predict(X_test_scaled)
    
    # Metrics
    lr_metrics = calculate_metrics(y_test, y_pred_lr, "Linear Regression")
    metrics_results.append(lr_metrics)
    
    # Plots
    plot_actual_vs_predicted(y_test, y_pred_lr, "Linear Regression")
    plot_residuals(y_test, y_pred_lr, "Linear Regression")
    
    # Feature Importance for Linear Regression (using absolute coefficient values)
    lr_importances = np.abs(lr_model.coef_)
    lr_importances = lr_importances / np.sum(lr_importances) # Normalize
    plot_feature_importance(lr_importances, feature_cols, "Linear Regression")
    
    # Save model
    lr_path = os.path.join("models", "linear_regression.joblib")
    joblib.dump(lr_model, lr_path)
    print(f"Saved Linear Regression model to {lr_path}")
    
    # ------------------ MODEL 2: RANDOM FOREST ------------------
    print("\n--- Training Model 2: Random Forest Regressor ---")
    # Using optimized hyperparameters to balance training speed and model depth
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_rf = rf_model.predict(X_test_scaled)
    
    # Metrics
    rf_metrics = calculate_metrics(y_test, y_pred_rf, "Random Forest")
    metrics_results.append(rf_metrics)
    
    # Plots
    plot_actual_vs_predicted(y_test, y_pred_rf, "Random Forest")
    plot_residuals(y_test, y_pred_rf, "Random Forest")
    
    # Feature Importance
    rf_importances = rf_model.feature_importances_
    plot_feature_importance(rf_importances, feature_cols, "Random Forest")
    
    # Save model
    rf_path = os.path.join("models", "random_forest.joblib")
    joblib.dump(rf_model, rf_path)
    print(f"Saved Random Forest model to {rf_path}")
    
    # ------------------ MODEL 3: XGBOOST REGRESSOR ------------------
    print("\n--- Training Model 3: XGBoost Regressor ---")
    xgb_model = XGBRegressor(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    
    # Metrics
    xgb_metrics = calculate_metrics(y_test, y_pred_xgb, "XGBoost")
    metrics_results.append(xgb_metrics)
    
    # Plots
    plot_actual_vs_predicted(y_test, y_pred_xgb, "XGBoost")
    plot_residuals(y_test, y_pred_xgb, "XGBoost")
    
    # Feature Importance
    xgb_importances = xgb_model.feature_importances_
    plot_feature_importance(xgb_importances, feature_cols, "XGBoost")
    
    # Save model
    xgb_path = os.path.join("models", "xgboost.joblib")
    joblib.dump(xgb_model, xgb_path)
    print(f"Saved XGBoost model to {xgb_path}")
    
    # ------------------ RESULTS SUMMARY ------------------
    print("\n=======================================================")
    print("               MODEL COMPARISON SUMMARY                ")
    print("=======================================================")
    summary_df = pd.DataFrame(metrics_results)
    print(summary_df.to_string(index=False))
    print("=======================================================")
    
    # Save summary to a CSV for reports/Streamlit
    summary_path = os.path.join("models", "model_comparison.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved comparison summary to {summary_path}")

if __name__ == "__main__":
    main()
