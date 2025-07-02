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


### Run

```bash
cd classification
python criticality_classifier.py
```
## Notes

- Input CSV must include features: `Time`, `Speed`, `Source`, `Name`, `Type`, `Master`, `Event_type`, `Real_Monitor`, `cod`, `Colour`, `Criticality`
- `Time` must be datetime-parsable

### Example Prediction Output

After training the model outputs (in model_outputs) a `predictions.json` file based on the test set. Each entry contains original input features along with the predicted probability and label. This json format ensures explainability and facilitates easy integration with dashboards.

```json
{
        "Time": "2023-09-18 17:18:46",
        "Source": "711-084",
        "Name": "710-170",
        "Criticality": 0,
        "Predicted_Probability": 0.467197448015213,
        "Predicted_Label": 0
}
