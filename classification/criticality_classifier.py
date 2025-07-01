import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from imblearn.combine import SMOTETomek

# Configurations
DATA_FILE = os.path.join("data", "TSR_040_DEF.csv")
OUTPUT_DIR = "model_output"
MODEL_FILE = os.path.join(OUTPUT_DIR, 'xgb_criticality_model.joblib')
METRICS_FILE = os.path.join(OUTPUT_DIR, "performance_metrics.json")
FEATURE_PLOT_FILE = os.path.join(OUTPUT_DIR, "feature_importance.png")
THRESHOLD_PLOT_FILE = os.path.join(OUTPUT_DIR, "threshold_analysis.png")
CUSTOM_THRESHOLD = 0.65

# --- Setup ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load Data ---
try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Error: {DATA_FILE} not found.")
    exit()

# --- Time-based Features ---
df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
df['hour_of_day'] = df['Time'].dt.hour
df['day_of_week'] = df['Time'].dt.dayofweek
df['month'] = df['Time'].dt.month
df['year'] = df['Time'].dt.year

# Train-Test Split 
train_df = df[df['year'].isin([2020, 2021, 2022])].copy()
test_df = df[df['year'].isin([2023])].copy()


numerical_columns = ['Speed', 'hour_of_day', 'day_of_week', 'month']
categorical_columns = ['Source', 'Name', 'Type', 'Master', 'Event_type', 'Real_Monitor', 'cod', 'Colour']

# Handle missing categorical values
for col in categorical_columns:
    train_df[col] = train_df[col].fillna('unknown').astype(str)
    test_df[col] = test_df[col].fillna('unknown').astype(str)

# --- One-Hot Encoding ---
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe.fit(train_df[categorical_columns])

X_train_cat = pd.DataFrame(ohe.transform(train_df[categorical_columns]),
                           columns=ohe.get_feature_names_out(),
                           index=train_df.index)
X_test_cat = pd.DataFrame(ohe.transform(test_df[categorical_columns]),
                          columns=ohe.get_feature_names_out(),
                          index=test_df.index)

X_train = pd.concat([X_train_cat, train_df[numerical_columns]], axis=1)
X_test = pd.concat([X_test_cat, test_df[numerical_columns]], axis=1)

y_train = train_df['Criticality']
y_test = test_df['Criticality']

# Save encoder for SHAP reuse
joblib.dump(ohe, os.path.join(OUTPUT_DIR, "encoder.joblib"))

# --- Rebalancing ---
print("Original class distribution:\n", y_train.value_counts())
smote = SMOTETomek(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("Resampled class distribution:\n", pd.Series(y_train_res).value_counts())

# --- Model Training ---
model = XGBClassifier(
    tree_method='hist',
    eval_metric='logloss',
    random_state=42,
    early_stopping_rounds=50,
    n_estimators=2000,
    eta=0.01,
)
model.fit(X_train_res, y_train_res,
          eval_set=[(X_train, y_train), (X_test, y_test)],
          verbose=False)

# --- Evaluation ---
y_prob = model.predict_proba(X_test)[:, 1]
y_pred_default = model.predict(X_test)
y_pred_custom = (y_prob >= CUSTOM_THRESHOLD).astype(int)

# Reports
print("\n--- Classification Report (Default Threshold) ---")
print(classification_report(y_test, y_pred_default))

print(f"\n--- Classification Report (Custom Threshold = {CUSTOM_THRESHOLD}) ---")
print(classification_report(y_test, y_pred_custom))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_custom))

# Save metrics
metrics = {
    "accuracy_default": accuracy_score(y_test, y_pred_default),
    "precision_default": precision_score(y_test, y_pred_default, zero_division=0),
    "recall_default": recall_score(y_test, y_pred_default, zero_division=0),
    "f1_default": f1_score(y_test, y_pred_default, zero_division=0),
    "accuracy_custom_threshold": accuracy_score(y_test, y_pred_custom),
    "precision_custom_threshold": precision_score(y_test, y_pred_custom, zero_division=0),
    "recall_custom_threshold": recall_score(y_test, y_pred_custom, zero_division=0),
    "f1_custom_threshold": f1_score(y_test, y_pred_custom, zero_division=0),
    "custom_threshold_value": CUSTOM_THRESHOLD,
    "confusion_matrix_custom_threshold": confusion_matrix(y_test, y_pred_custom).tolist(),
    "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}
with open(METRICS_FILE, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"Metrics saved to {METRICS_FILE}")

# --- Feature Importance Plot ---
feature_importance = pd.Series(model.feature_importances_, index=X_train.columns)
top_features = feature_importance.sort_values(ascending=False).head(30)

plt.figure(figsize=(10, 8))
top_features.plot(kind='barh')
plt.gca().invert_yaxis()
plt.title("Top 30 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(FEATURE_PLOT_FILE)
plt.close()
print(f"Feature importance plot saved to {FEATURE_PLOT_FILE}")

# --- Threshold Analysis Plot ---
thresholds = np.arange(0.0, 1.01, 0.01)
precisions = []
recalls = []
f1s = []

for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    precisions.append(precision_score(y_test, y_pred_t, zero_division=0))
    recalls.append(recall_score(y_test, y_pred_t, zero_division=0))
    f1s.append(f1_score(y_test, y_pred_t, zero_division=0))

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions, label='Precision')
plt.plot(thresholds, recalls, label='Recall')
plt.plot(thresholds, f1s, label='F1 Score')
plt.axvline(CUSTOM_THRESHOLD, color='gray', linestyle='--', label=f'Threshold = {CUSTOM_THRESHOLD}')
plt.title("Precision, Recall, F1 vs Threshold")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(THRESHOLD_PLOT_FILE)
plt.close()
print(f"Threshold analysis plot saved to {THRESHOLD_PLOT_FILE}")

# --- Save Model ---
joblib.dump(model, MODEL_FILE)
print(f"Trained model saved to {MODEL_FILE}")
