# Time Series Forecasting of Critical Alarms

This script automates the process of forecasting critical railway alarms (`Criticality_1`) for individual train cars using time-series analysis with the **Prophet** library.

---

## Input

The script requires a single CSV data file as input.

- **File Format:** CSV file containing historical alarm data  
- **Default Path:** `data/TSR_040_daily.csv`  
- **Required Columns:**
  - `ts`: Timestamp of the data entry (date or datetime)
  - `Name_mode`: Unique identifier for each train car (e.g., `Vagone_1`)
  - `Criticality_1`: Count of critical alarms
  - `Criticality_0`: Count of non-critical alarms

---

## Outputs

The script generates several output files in the specified output directory:

- `railway_predictions.log`: A log file containing detailed information about the script's execution, including processed trains, warnings, and errors.

- `railway_analysis.json`: A comprehensive JSON file containing:
  - Historical statistics (mean, total, max, etc.) for each train.
  - Daily forecast values for the specified forecast period.
  - Comparison metrics (historical vs. predicted).
  - Model performance metrics (MAE, RMSE, MAPE).

- `railway_summary_predictions.png`: An image file with two summary heatmaps:
  - Forecast Heatmap: Average predicted alarms per day of the week.
  - Comparison Heatmap: Historical average vs. predicted average alarms.

- `railway_7day_forecast.png`: A heatmap image showing the daily predicted number of critical alarms for the next 7 days for each train.

- `railway_report.csv`: A CSV file summarizing the comparison between historical and predicted alarm averages for each train.

- `train_plots/` (Directory): A folder containing individual forecast plots for each train car. Each plot (`forecast_Vagone_X.png`) visualizes:
  - Historical alarm data.
  - The model's forecast.
  - The 95% confidence interval.
  - Key performance metrics.

---

## How to Run

```bash
python timeseries_analysis.py
