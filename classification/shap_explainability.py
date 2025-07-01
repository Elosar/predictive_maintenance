import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
RUN_SHAP = True
OUTPUT_DIR = "model_output"
MODEL_FILE = os.path.join(OUTPUT_DIR, "xgb_criticality_model.joblib")
DATA_FILE = os.path.join("data", "TSR_040_DEF.csv")

# --- LOAD DATA ---
df = pd.read_csv(DATA_FILE)

# Time features
df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
df['hour_of_day'] = df['Time'].dt.hour
df['day_of_week'] = df['Time'].dt.dayofweek
df['month'] = df['Time'].dt.month
df['year'] = df['Time'].dt.year

# Split data for consistency with training
df = df[df['year'].isin([2023])].copy()

# Define columns
numerical_columns = ['Speed', 'hour_of_day', 'day_of_week', 'month']
categorical_columns = ['Source', 'Name', 'Type', 'Master', 'Event_type', 'Real_Monitor', 'cod', 'Colour']

for col in categorical_columns:
    df[col] = df[col].fillna('unknown').astype(str)

# Load the encoder used during training
encoder_path = os.path.join(OUTPUT_DIR, "encoder.joblib")
if os.path.exists(encoder_path):
    ohe = joblib.load(encoder_path)
else:
    raise FileNotFoundError("You need to save the encoder in training script to use it here.")

X_cat = pd.DataFrame(ohe.transform(df[categorical_columns]),
                     columns=ohe.get_feature_names_out(),
                     index=df.index)
X = pd.concat([X_cat, df[numerical_columns]], axis=1)

# --- LOAD MODEL ---
model = joblib.load(MODEL_FILE)

# --- SHAP ANALYSIS ---
if RUN_SHAP:
    print("Running SHAP analysis...")

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_plot.png"))
    print("SHAP summary plot saved.")

    # Optional: Top feature dependence plot
    top_feature = X.columns[shap_values.abs.mean(0).values.argmax()]
    plt.figure()
    shap.dependence_plot(top_feature, shap_values.values, X, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"shap_dependence_{top_feature}.png"))
    print(f"SHAP dependence plot for '{top_feature}' saved.")
else:
    print("SHAP execution skipped. Set RUN_SHAP = True to run it.")
