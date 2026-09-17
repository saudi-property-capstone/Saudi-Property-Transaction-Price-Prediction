import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Setup output directories for professional reporting
os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

print("Loading data for Ridge Regression baseline evaluation...")

# Load scaled training and validation features and targets
X_train = np.load("data/processed/X_train_scaled.npy")
y_train = pd.read_csv("data/processed/y_train_log.csv.gz")['log_price'].values

X_val = np.load("data/processed/X_validation_scaled.npy")
y_val = pd.read_csv("data/processed/y_validation_log.csv.gz")['log_price'].values

# Train Ridge Regression baseline model
print("Training Ridge Regression baseline model...")
ridge_baseline = Ridge(alpha=1.0, random_state=42)
ridge_baseline.fit(X_train, y_train)

# Predict and evaluate baseline performance
y_pred_ridge = ridge_baseline.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, y_pred_ridge))
mae = mean_absolute_error(y_val, y_pred_ridge)
r2 = r2_score(y_val, y_pred_ridge)

print("=" * 60)
print("RIDGE REGRESSION BASELINE EVALUATION REPORT")
print("=" * 60)
print(f"Root Mean Squared Error (RMSE) : {rmse:.4f}")
print(f"Mean Absolute Error (MAE)      : {mae:.4f}")
print(f"R² Score                       : {r2:.4f}")
print("=" * 60)

# Save evaluation metrics to a summary CSV table
metrics_df = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'R2_Score'],
    'Value': [rmse, mae, r2]
})
metrics_df.to_csv("outputs/tables/historical_baseline_metrics.csv", index=False)
print("Metrics successfully saved to outputs/tables/historical_baseline_metrics.csv")

# Generate professional multi-panel visual charts
print("Generating advanced multi-panel evaluation plots...")
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Actual vs Predicted Scatter Plot with Identity Line
sns.scatterplot(x=y_val, y=y_pred_ridge, alpha=0.3, ax=axes[0], color="royalblue")
axes[0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], '--r', linewidth=2, label="Perfect Prediction (1:1)")
axes[0].set_title("Actual vs Predicted (Ridge Baseline)", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Actual Log Price", fontsize=10)
axes[0].set_ylabel("Predicted Log Price", fontsize=10)
axes[0].legend(loc="upper left")

# Subplot 2: Residuals Distribution with KDE
residuals = y_val - y_pred_ridge
sns.histplot(residuals, kde=True, ax=axes[1], color="forestgreen", bins=50)
axes[1].set_title("Baseline Residuals Distribution", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Residual (Actual - Predicted)", fontsize=10)
axes[1].set_ylabel("Frequency", fontsize=10)

plt.tight_layout()
plot_path = "outputs/figures/historical_baseline_evaluation.png"
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"Professional multi-panel evaluation plots saved to {plot_path}")
print("Ridge Baseline pipeline execution completed successfully!")