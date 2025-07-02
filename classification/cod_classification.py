import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import joblib
import os

# --- Configuration ---
OUTPUT_DIR = "model_artifacts_cod"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load and filter dataset ---
df = pd.read_csv("TSR_040_DEF.csv")
df = df.dropna(subset=['cod'])                  # Drop missing targets
df = df[df['cod'].isin([2, 5])]                 # Keep only desired cod classes

# --- Time features ---
df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
df = df.dropna(subset=['Time'])                # Drop bad time rows
df['hour_of_day'] = df['Time'].dt.hour
df['day_of_week'] = df['Time'].dt.dayofweek
df['month'] = df['Time'].dt.month
df['year'] = df['Time'].dt.year

# --- Train/Test Split by year ---
train_df = df[df['year'].isin([2020, 2021, 2022])].copy()
test_df  = df[df['year'] == 2023].copy()

# --- Feature Setup ---
numerical_columns = ['Speed', 'hour_of_day', 'day_of_week', 'month']
categorical_columns = ['Source', 'Name', 'Type', 'Master', 'Event_type', 'Real_Monitor','id','id1']

# Ensure string type for categorical
for col in categorical_columns:
    train_df[col] = train_df[col].fillna('unknown').astype(str)
    test_df[col] = test_df[col].fillna('unknown').astype(str)

# --- OneHotEncode Categorical Features ---
ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe_encoder.fit(train_df[categorical_columns])
ohe_feature_names = ohe_encoder.get_feature_names_out(categorical_columns)

X_train_cat = pd.DataFrame(
    ohe_encoder.transform(train_df[categorical_columns]),
    columns=ohe_feature_names,
    index=train_df.index
)
X_test_cat = pd.DataFrame(
    ohe_encoder.transform(test_df[categorical_columns]),
    columns=ohe_feature_names,
    index=test_df.index
)

# --- Final Feature Matrices ---
X_train = pd.concat([X_train_cat, train_df[numerical_columns]], axis=1)
X_test  = pd.concat([X_test_cat,  test_df[numerical_columns]], axis=1)

# --- Target Variable ---
y_train = train_df['cod'].astype(str)
y_test  = test_df['cod'].astype(str)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded  = label_encoder.transform(y_test)

# --- Model Training ---
model_cod = XGBClassifier(
    objective='multi:softprob',
    num_class=len(label_encoder.classes_),
    tree_method='hist',
    eval_metric='mlogloss',
    random_state=42,
    n_estimators=1000,
    early_stopping_rounds=50,
    learning_rate=0.01
)

model_cod.fit(X_train, y_train_encoded,
              eval_set=[(X_test, y_test_encoded)],
              verbose=True)

# --- Evaluation ---
y_pred_encoded = model_cod.predict(X_test)
y_pred_labels = np.argmax(y_pred_encoded, axis=1)  # pick highest-prob class
y_pred = label_encoder.inverse_transform(y_pred_labels)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Save Artifacts ---
joblib.dump(model_cod, os.path.join(OUTPUT_DIR, "xgb_model_cod.pkl"))
joblib.dump(ohe_encoder, os.path.join(OUTPUT_DIR, "ohe_encoder_cod.pkl"))
joblib.dump(ohe_feature_names, os.path.join(OUTPUT_DIR, "ohe_feature_names_cod.pkl"))
joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, "label_encoder_cod.pkl"))

print("\nModel and encoders saved.")
