"""
EDA (Exploratory Data Analysis) - Air Quality & PM2.5 Prediction Project
COM2502 Introduction to Data Science

This script generates all required EDA visualizations including:
  1. Summary statistics
  2. PM2.5 distribution plot
  3. Feature distributions
  4. Correlation heatmap
  5. PM2.5 vs key weather features scatter plots
  6. PM2.5 by season and day-of-week boxplots
  7. Continent-level PM2.5 comparison
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

# ── Output directory ──────────────────────────────────────────────────────────
os.makedirs("plots/eda", exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ── Load Data ─────────────────────────────────────────────────────────────────
print("Loading cleaned_data.csv...")
df = pd.read_csv("cleaned_data.csv")
print(f"Dataset shape: {df.shape}")

# Numeric feature columns used in modeling
FEATURE_COLS = [
    "temp", "feels_like", "pressure", "humidity", "wind_speed",
    "clouds_all", "visibility", "city_population", "pm2_5"
]

# ─────────────────────────────────────────────────────────────────────────────
# 1. Summary Statistics
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Summary Statistics ---")
summary = df[FEATURE_COLS].describe().round(3)
print(summary.to_string())
summary.to_csv("plots/eda/summary_statistics.csv")
print("Saved: plots/eda/summary_statistics.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PM2.5 Distribution
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df["pm2_5"], bins=60, color="#3b82f6", edgecolor="white", alpha=0.85)
axes[0].axvline(df["pm2_5"].mean(),  color="#ef4444", linestyle="--", lw=2,
                label=f"Mean = {df['pm2_5'].mean():.2f} µg/m³")
axes[0].axvline(df["pm2_5"].median(), color="#22c55e", linestyle="--", lw=2,
                label=f"Median = {df['pm2_5'].median():.2f} µg/m³")
axes[0].set_title("PM2.5 Distribution (after IQR Outlier Removal)")
axes[0].set_xlabel("PM2.5 Concentration (µg/m³)")
axes[0].set_ylabel("Frequency")
axes[0].legend()

sns.kdeplot(df["pm2_5"], ax=axes[1], fill=True, color="#6366f1", alpha=0.6)
axes[1].set_title("PM2.5 Density Curve")
axes[1].set_xlabel("PM2.5 Concentration (µg/m³)")
axes[1].set_ylabel("Density")

plt.suptitle("Figure 1: PM2.5 Target Variable Distribution", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("plots/eda/pm25_distribution.png", bbox_inches="tight")
plt.close()
print("Saved: plots/eda/pm25_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature Distributions (Grid)
# ─────────────────────────────────────────────────────────────────────────────
plot_features = ["temp", "humidity", "wind_speed", "pressure", "clouds_all", "visibility"]
colors = ["#f59e0b", "#3b82f6", "#10b981", "#8b5cf6", "#ef4444", "#06b6d4"]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for i, (feat, color) in enumerate(zip(plot_features, colors)):
    axes[i].hist(df[feat].dropna(), bins=50, color=color, edgecolor="white", alpha=0.8)
    axes[i].set_title(feat.replace("_", " ").title())
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Count")

plt.suptitle("Figure 2: Distribution of Key Weather Features", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/eda/feature_distributions.png", bbox_inches="tight")
plt.close()
print("Saved: plots/eda/feature_distributions.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────
corr_cols = ["pm2_5", "temp", "humidity", "wind_speed", "pressure",
             "clouds_all", "visibility", "city_population", "is_weekend",
             "month", "hour", "day_of_week"]

corr_matrix = df[corr_cols].corr().round(2)

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdYlBu_r",
    center=0,
    linewidths=0.5,
    ax=ax,
    square=True,
    cbar_kws={"shrink": 0.8}
)
ax.set_title("Figure 3: Feature Correlation Matrix (Lower Triangle)", fontsize=15, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("plots/eda/correlation_heatmap.png", bbox_inches="tight")
plt.close()
print("Saved: plots/eda/correlation_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. PM2.5 vs Key Weather Features (Scatter Grid)
# ─────────────────────────────────────────────────────────────────────────────
scatter_features = [
    ("temp",           "Temperature (°C)"),
    ("humidity",       "Humidity (%)"),
    ("wind_speed",     "Wind Speed (m/s)"),
    ("pressure",       "Pressure (hPa)"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
scatter_colors = ["#f59e0b", "#3b82f6", "#10b981", "#8b5cf6"]

sample = df.sample(min(3000, len(df)), random_state=42)  # sample for readability

for i, ((feat, label), color) in enumerate(zip(scatter_features, scatter_colors)):
    axes[i].scatter(sample[feat], sample["pm2_5"], alpha=0.15, s=10, color=color)
    # Trend line
    m, b = np.polyfit(sample[feat].dropna(), sample.loc[sample[feat].notna(), "pm2_5"], 1)
    x_line = np.linspace(sample[feat].min(), sample[feat].max(), 200)
    axes[i].plot(x_line, m * x_line + b, color="black", lw=2, label=f"Trend (slope={m:.3f})")
    axes[i].set_xlabel(label)
    axes[i].set_ylabel("PM2.5 (µg/m³)")
    axes[i].set_title(f"PM2.5 vs {label}")
    axes[i].legend(fontsize=9)

plt.suptitle("Figure 4: PM2.5 vs Key Meteorological Features", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/eda/pm25_vs_features.png", bbox_inches="tight")
plt.close()
print("Saved: plots/eda/pm25_vs_features.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. PM2.5 by Season and Day-of-Week (Boxplots)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Season order
season_order = ["spring", "summer", "autumn", "winter"]
available_seasons = [s for s in season_order if s in df["season"].unique()]
season_palette = {"spring": "#22c55e", "summer": "#f59e0b", "autumn": "#f97316", "winter": "#3b82f6"}

sns.boxplot(
    data=df, x="season", y="pm2_5", order=available_seasons,
    palette={k: v for k, v in season_palette.items() if k in available_seasons},
    ax=axes[0]
)
axes[0].set_title("PM2.5 Concentration by Season")
axes[0].set_xlabel("Season")
axes[0].set_ylabel("PM2.5 (µg/m³)")

day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
sns.boxplot(
    data=df, x="day_of_week", y="pm2_5",
    palette="muted", ax=axes[1]
)
axes[1].set_xticklabels(day_labels)
axes[1].set_title("PM2.5 Concentration by Day of Week")
axes[1].set_xlabel("Day of Week")
axes[1].set_ylabel("PM2.5 (µg/m³)")

plt.suptitle("Figure 5: PM2.5 Variation by Season and Weekday", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/eda/pm25_by_season_weekday.png", bbox_inches="tight")
plt.close()
print("Saved: plots/eda/pm25_by_season_weekday.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. PM2.5 by Continent (Barplot — Mean ± Std)
# ─────────────────────────────────────────────────────────────────────────────
continent_stats = (
    df.groupby("continent")["pm2_5"]
    .agg(mean="mean", std="std")
    .reset_index()
    .sort_values("mean", ascending=False)
)

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(
    continent_stats["continent"],
    continent_stats["mean"],
    yerr=continent_stats["std"],
    capsize=5,
    color=sns.color_palette("Set2", len(continent_stats)),
    edgecolor="white",
    alpha=0.9
)
ax.set_title("Figure 6: Mean PM2.5 Concentration by Continent", fontsize=15, fontweight="bold")
ax.set_xlabel("Continent")
ax.set_ylabel("Mean PM2.5 (µg/m³)")
ax.axhline(df["pm2_5"].mean(), color="red", linestyle="--", lw=1.5,
           label=f"Global Mean = {df['pm2_5'].mean():.2f} µg/m³")
ax.legend()
plt.tight_layout()
plt.savefig("plots/eda/pm25_by_continent.png", bbox_inches="tight")
plt.close()
print("Saved: plots/eda/pm25_by_continent.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. PM2.5 by Hour of Day (Line plot)
# ─────────────────────────────────────────────────────────────────────────────
hourly = df.groupby("hour")["pm2_5"].mean()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(hourly.index, hourly.values, marker="o", color="#6366f1", lw=2.5, markersize=7)
ax.fill_between(hourly.index, hourly.values, alpha=0.15, color="#6366f1")
ax.set_title("Figure 7: Average PM2.5 Concentration by Hour of Day", fontsize=15, fontweight="bold")
ax.set_xlabel("Hour of Day (0 = Midnight, 12 = Noon)")
ax.set_ylabel("Mean PM2.5 (µg/m³)")
ax.set_xticks(range(0, 24))
ax.grid(True, linestyle=":", alpha=0.7)
plt.tight_layout()
plt.savefig("plots/eda/pm25_by_hour.png", bbox_inches="tight")
plt.close()
print("Saved: plots/eda/pm25_by_hour.png")

print("\n✅ All EDA plots saved to plots/eda/")
