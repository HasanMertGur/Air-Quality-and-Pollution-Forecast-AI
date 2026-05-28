import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def calculate_metrics(y_true, y_pred, model_name):
    """
    Calculate and return key regression metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"[{model_name}] Evaluation Metrics:")
    print(f"  - Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  - Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  - R-squared (R2) Score: {r2:.4f}\n")
    
    return {"Model": model_name, "MAE": mae, "RMSE": rmse, "R2": r2}

def plot_actual_vs_predicted(y_true, y_pred, model_name, save_dir="plots"):
    """
    Generate and save a scatter plot of Actual vs Predicted values.
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    
    # Scatter plot with regression line indicator
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.4, color="#3498db")
    
    # Perfect prediction line
    max_val = max(max(y_true), max(y_pred))
    min_val = min(min(y_true), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color="#e74c3c", linestyle="--", lw=2, label="Perfect Prediction")
    
    plt.title(f"{model_name}: Actual vs. Predicted PM2.5", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Actual PM2.5 (µg/m³)", fontsize=12)
    plt.ylabel("Predicted PM2.5 (µg/m³)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_actual_vs_predicted.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot: {plot_path}")

def plot_feature_importance(importances, feature_names, model_name, save_dir="plots"):
    """
    Generate and save a feature importance bar chart.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis")
    
    plt.title(f"{model_name}: Feature Importance", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Relative Importance Score", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.grid(True, axis='x', linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_feature_importance.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot: {plot_path}")

def plot_residuals(y_true, y_pred, model_name, save_dir="plots"):
    """
    Generate and save a residuals plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    residuals = y_true - y_pred
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.4, color="#9b59b6")
    plt.axhline(y=0, color="#2c3e50", linestyle="-", lw=1.5)
    
    plt.title(f"{model_name}: Residuals Analysis", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted PM2.5 (µg/m³)", fontsize=12)
    plt.ylabel("Residuals (Actual - Predicted)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_residuals.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot: {plot_path}")
