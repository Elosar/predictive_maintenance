# Criticality Classification Pipeline

This pipeline predicts whether an alarm in train diagnostic data is **critical (1)** or **not critical (0)** using an XGBoost classifier. It also provides **SHAP-based explainability** in a separate script.

Objective is to predict the Criticality class (binary classification). Overview:

- **Preprocessing:** Time features, One-Hot Encoding
- **Train/Test Split:** 2020–2022 for training, 2023 for testing
- **Class Balancing:** SMOTETomek
- **Model:** XGBoost (`tree_method='hist'`, early stopping, 2000 trees)
- **Evaluation:** Accuracy, Precision, Recall, F1 at thresholds 0.5 and 0.65
- **Visuals:** Feature importance + threshold-performance trade-offs
---
## How to Run

### Install Dependencies using

pip install -r requirements.txt


### Train & Evaluate Model

```bash
cd classification
python criticality_classifier.py

## Notes

- Input CSV must include features: `Time`, `Speed`, `Source`, `Name`, `Type`, `Master`, `Event_type`, `Real_Monitor`, `cod`, `Colour`, `Criticality`
- `Time` must be datetime-parsable