# EcoCast: Air Quality & PM2.5 Regression Prediction Pipeline

This repository contains the complete end-to-end data science pipeline developed for **COM2502: Introduction to Data Science** Course Project. The objective is to perform numerical regression to predict daily fine particulate matter (**$PM_{2.5}$**) concentrations using meteorological elements and urban activity proxies.

---

## 📂 Project Structure

```text
air_quality_prediction/
├── raw_data.csv                # Raw dataset from OpenWeather (20,419 rows)
├── cleaned_data.csv            # Cleaned dataset (processed by Data Engineer)
├── models/
│   ├── linear_regression.joblib    # Baseline linear regressor
│   ├── random_forest.joblib        # Ensemble bagging regressor
│   ├── xgboost.joblib              # Gradient boosting regressor
│   ├── scaler.joblib               # StandardScaler binary
│   └── model_comparison.csv        # Metrics summary table
├── src/
│   ├── clean_data.py               # Data cleaning and outlier removal script
│   ├── train.py                    # Preprocessing, training, and metrics pipeline
│   ├── utils.py                    # Metric calculators and matplotlib/seaborn plot generators
│   └── create_notebook.py          # Script generating model_training.ipynb
├── notebooks/
│   └── model_training.ipynb        # Fully-annotated Jupyter Notebook for report/submission
├── plots/
│   ├── xgboost_actual_vs_predicted.png
│   ├── xgboost_feature_importance.png
│   └── xgboost_residuals.png
├── dashboard/
│   └── app.py                      # Premium Streamlit presentation dashboard
├── .gitignore                  # Prevents committing venv and cache
└── README.md                   # Project overview and execution manual
```

---

## 📊 Dataset & Pipeline Overview

### 1. Data Cleaning (Data Engineer — Person 1)
* **Source**: OpenWeather API, gathering readings from the top 3 most populous cities per country globally (representing diverse climates and populations).
* **Cleaning Workflow (`src/clean_data.py`)**:
  * Removed duplicate rows.
  * Applied forward-fill (`ffill`) for minor meteorological gaps.
  * Imputed minor missing features (e.g. `visibility`) with their column medians.
  * **Outlier Filtering**: Applied **Interquartile Range (IQR)** outlier detection on the target variable ($PM_{2.5}$). Sensor anomalies exceeding $Q_3 + 1.5 \times IQR$ were discarded, reducing the dataset to a highly reliable **18,096 records**.

### 2. Machine Learning Regression (Model Architect — Person 2)
* **Feature Selection**: Trained on 12 continuous and temporal predictors:
  * *Meteorology*: `temp` (Temperature), `feels_like`, `pressure` (Atmospheric Pressure), `humidity`, `wind_speed`, `clouds_all`, `visibility`.
  * *Urban Activity*: `city_population` (highly effective proxy for local traffic density and industrial emissions).
  * *Temporal/Cyclic*: `is_weekend`, `month`, `hour`, `day_of_week`.
* **Preprocessing**: Applied standard scaling (`StandardScaler`) to features and performed an **80-20 Train-Test Split** (14,476 training samples / 3,620 testing samples).
* **Models Trained**:
  * **Linear Regression**: A standard statistical baseline to check for linear coefficients.
  * **Random Forest Regressor**: An ensemble bagging model training 100 decision trees to map non-linear physical weather states.
  * **XGBoost Regressor**: An advanced gradient-boosted decision tree algorithm optimized for tabular data.

### 3. Interactive Web Dashboard (Viz & Bonus Hunter — Person 3)
* A high-fidelity, responsive **Streamlit Web Application** designed for the live presentation demonstration.
* **Sliders**: Real-time slide controls to alter temperature, humidity, wind speed, pressure, and local population.
* **Outputs**: Displays instant PM2.5 predictions for all 3 models side-by-side, color-coded health warning badges (Good, Moderate, Poor, Hazardous) based on official WHO/EPA metrics, and direct model diagnostics tabs (Residuals, Feature Importances, Actual vs. Predicted scatter plots).

---

## 📈 Model Performance Comparison

Evaluating our models on the holdout test set yielded a massive **"Data Cleaning Dividend."** By utilizing the IQR outlier filtering, **MAE dropped by over 50%** compared to the uncleaned dataset:

| Model | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | R-squared ($R^2$) Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression** (Baseline) | 3.7935 µg/m³ | 5.0391 µg/m³ | 0.1202 |
| **Random Forest** (Ensemble) | 2.9393 µg/m³ | 4.1525 µg/m³ | 0.4026 |
| **XGBoost** (Gradient Boosting) | **2.8639 µg/m³** | **4.0586 µg/m³** | **0.4293** |

### Key Insights:
1. **Atmospheric Complexity**: The baseline linear model struggles ($R^2 \approx 0.12$), showing that air quality has non-linear meteorological mechanics.
2. **Top Performers**: **XGBoost** is our champion model ($R^2 = 0.4293$), accurately predicting PM2.5 concentrations within an average of **2.86 µg/m³** from actual values.
3. **Scientific Observations**:
   * **Humidity & Temperature**: Highly significant non-linear predictors of suspended particulate mass.
   * **Wind Speed**: Correlated inversely with PM2.5, proving the physical dispersal effect mathematically.
   * **City Population**: Extremely strong positive feature importance, demonstrating that urban activity/traffic density directly drives particulate spikes.

---

## 🚀 Quick Start Guide

### 1. Installation & Environment Setup
Clone the repository or open your VS Code terminal inside the project directory, then execute the following commands to initialize the virtual environment and install dependencies:

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install requirements
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib streamlit pyarrow
```

### 2. Executing the Preprocessing and Modeling
To perform data cleaning, retrain the models, export the joblib files, and generate evaluation plots, run:

```powershell
# Run data cleaning (Person 1 script)
python src/clean_data.py

# Run model training and export (Person 2 script)
python src/train.py

# Generate the Jupyter Notebook
python src/create_notebook.py
```

### 3. Launching the Interactive Streamlit App
To run the live demo server locally:

```powershell
streamlit run dashboard/app.py
```
This launches a local web server (usually at `http://localhost:8501`) displaying the interactive forecasting dashboard.
