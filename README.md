# 🚆 Predictive Maintenance & Alarm Intelligence for Railway Transport

This project builds a data-driven system for **predictive maintenance** and **anomaly detection** in railway transport, focusing on the **train’s traction system**. It enables:

- Early detection of abnormal alarm behavior
- Classification of **critical alarms**
- Forecasting of alarm volumes using time series analysis

It combines **machine learning**, **time series modeling**, and **explainable AI (XAI)** techniques, applied to Trenord telemetry and maintenance data.

---

## 📌 About the Project

The project explores the feasibility of applying a **data-driven strategy** within the decision-making process of **Trenord**, a regional railway operator in Northern Italy. It leverages historical alarm data, maintenance logs, and service information to build machine learning and time series models for early warning and classification systems.

---

## 📊 Dataset

The dataset was collected and pre-processed as part of a predictive maintenance initiative at the **University of Bologna** in collaboration with **Trenord** (Giulia Millitarì, 2023/2024). This project extends and adapts that work to build new tools.

### 🔹 Dataset Files

| File              | Description |
|------------------|-------------|
| `TSR_040_DEF.csv` | Event-level alarm dataset with 101,708 rows × 19 columns. Contains timestamps, sources, alarm codes, GPS, wagon IDs, and labels like `Criticality`. |
| `TSR_040_daily.csv` | Daily aggregated dataset with 767 rows × 259 columns. Includes alert counts, maintenance/service data, and total km traveled. |

## 🧠 Modules Overview

This project contains two main components:

### 🔹 1. Alarm Classification

Predict whether an individual alarm is **critical** or **non-critical** using an XGBoost model.  
Includes class rebalancing (SMOTETomek), categorical encoding, threshold analysis, and SHAP explanations.

📘 Detailed instructions: [classification/README.md](classification/README.md)

---

### 🔹 2. Time Series Forecasting

Forecast the **number of daily critical alarms** per train car using Prophet.  
Generates individual forecasts, heatmaps, and performance comparisons.

📘 Detailed instructions: [time_series/README.md](timeseries_forecasting/README.md)

## ▶️ Run the Project

### Install required dependencies:

pip install -r requirements.txt

To run classificatio pipeline

cd classification
python criticality_classifier.py         # Trains and saves model, metrics, plots
python shap_explanations.py              # Optional: generates SHAP plots

To run Time Series Pipeline

cd timeseries
python timeseries_analysis.py

## 📁 Outputs Summary

Each pipeline saves outputs to its module directory:

### 🔹 Classification (`classification/model_output/`)
- Trained model (`xgb_criticality_model.joblib`)
- Metrics file (`performance_metrics.json`)
- Feature importance plot
- Threshold tuning plot
- SHAP summary and dependence plots (optional)

### 🔹 Forecasting (`timeseries_forecasting/output/`)
- Forecast plots per train (`train_plots/`)
- 7-day forecast heatmap
- Weekly average comparison heatmap
- Full analysis JSON (`railway_analysis.json`)
- Summary report CSV
- Logs (`railway_predictions.log`)

---

## 🚧 Roadmap

- [ ] Automate pipeline for real-time streaming data  
- [ ] Add LSTM-based forecasting for time series  
- [ ] Integration to a monitoring dashboard for real-time insights
- [ ] Exploring and anaylising alternative preprocessing of diagnostic logs for improved performance    

---

## 🙋‍♀️ Credits
- **Lutech SpA - Smart Mobility**
- **Trenord**
- **University of Bologna – Department of Statistical Sciences**  
  _Based on the thesis and dataset by Giulia Millitarì (2023/2024)_

