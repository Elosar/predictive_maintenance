#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alarm Prediction System
Automated script for predicting Criticality_1 alarms on railway vehicles

"""

# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import sys
import warnings
from prophet import Prophet
import argparse
import logging
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('railway_predictions.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RailwayAlarmPredictor:
    """
    Main class for railway alarm predictions
    """

    def __init__(self, file_path, forecast_days=30):
        self.file_path = file_path
        self.forecast_days = forecast_days
        self.df = None
        self.df_filtered = None
        self.all_predictions = {}
        self.performance_metrics = {}

    def load_and_preprocess_data(self):
        """Load and preprocess the CSV data"""
        logger.info(f"Loading data from {self.file_path}")

        try:
            # Load data
            self.df = pd.read_csv(self.file_path, delimiter=',')

            # Rename columns
            self.df = self.df.rename(columns={
                'Source': 'source', 'Name': 'name', 'Time': 'ts', 'Event_type': 'event_type',
                'Type': 'alert_type', 'Real_Monitor': 'machine_type', 'cod': 'cod', 'id': 'id',
                'id1': 'id1', 'Complete_code': 'complete_code', 'Master': 'master', 'Speed': 'speed',
                'Depot': 'depot', 'Lat': 'lat', 'Lon': 'lon', 'Lat_umt': 'lat_umt', 'Lon_umt': 'lon_umt',
                'Criticality': 'criticality', 'Colour': 'colour'
            })
            
            # Filter relevant features - assuming 'Name_mode', 'Criticality_0', 'Criticality_1' exist
            features = ['ts', 'Name_mode', 'Criticality_0', 'Criticality_1']
            self.df['ts'] = pd.to_datetime(self.df['ts'])
            self.df_filtered = self.df[features].copy()

            # Add derived features
            self.df_filtered['Total_Alarms'] = self.df_filtered['Criticality_0'] + self.df_filtered['Criticality_1']
            self.df_filtered = self.df_filtered.sort_values(['Name_mode', 'ts']).reset_index(drop=True)

            logger.info(f"Data loaded successfully. Shape: {self.df_filtered.shape}")
            logger.info(f"Date range: {self.df_filtered['ts'].min()} to {self.df_filtered['ts'].max()}")
            logger.info(f"Number of trains: {self.df_filtered['Name_mode'].nunique()}")

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def calculate_performance_metrics(self, train_data, vagone):
        """Calculate performance metrics using backtesting"""
        df_prophet = train_data[['ts', 'Criticality_1']].rename(columns={'ts': 'ds', 'Criticality_1': 'y'})
        
        if len(df_prophet) < 30:  # Need enough data for meaningful split
            return None
        
        # Split data: 80% train, 20% test
        split_point = int(len(df_prophet) * 0.8)
        train_split = df_prophet[:split_point]
        test_split = df_prophet[split_point:]
        
        try:
            # Train model on training split
            model = Prophet(
                yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False,
                changepoint_prior_scale=0.3, seasonality_prior_scale=10, interval_width=0.95
            )
            model.fit(train_split)
            
            # Predict on test split
            future = model.make_future_dataframe(periods=len(test_split), freq='D')
            forecast = model.predict(future)
            
            # Get test predictions
            test_predictions = forecast.tail(len(test_split))['yhat'].clip(lower=0)
            actual_values = test_split['y'].values
            
            # Calculate metrics
            mae = mean_absolute_error(actual_values, test_predictions)
            rmse = np.sqrt(mean_squared_error(actual_values, test_predictions))
            
            # MAPE calculation with zero handling
            non_zero_mask = actual_values != 0
            if non_zero_mask.sum() > 0:
                mape = np.mean(np.abs((actual_values[non_zero_mask] - test_predictions[non_zero_mask]) / actual_values[non_zero_mask])) * 100
            else:
                mape = 0
            
            return {
                'mae': round(mae, 3),
                'rmse': round(rmse, 3),
                'mape': round(mape, 2)
            }
            
        except Exception as e:
            logger.warning(f"Could not calculate metrics for {vagone}: {str(e)}")
            return None

    def predict_all_trains(self):
        """Generate predictions for all trains"""
        logger.info(f"Starting predictions for all trains...")

        for vagone in self.df_filtered['Name_mode'].unique():
            logger.info(f"Processing train: {vagone}")

            vagone_data = self.df_filtered[self.df_filtered['Name_mode'] == vagone].copy().sort_values('ts')

            if len(vagone_data) < 10:
                logger.warning(f"Insufficient data for {vagone} (only {len(vagone_data)} records)")
                continue

            df_prophet = vagone_data[['ts', 'Criticality_1']].rename(columns={'ts': 'ds', 'Criticality_1': 'y'})

            if df_prophet['y'].std() < 0.1:
                logger.warning(f"{vagone}: Data too constant for predictions")
                continue

            try:
                # Calculate performance metrics first
                metrics = self.calculate_performance_metrics(vagone_data, vagone)
                if metrics:
                    self.performance_metrics[vagone] = metrics
                
                # Train final model on all data
                model = Prophet(
                    yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False,
                    changepoint_prior_scale=0.3, seasonality_prior_scale=10, interval_width=0.95
                )
                model.fit(df_prophet)
                future = model.make_future_dataframe(periods=self.forecast_days)
                forecast = model.predict(future)

                forecast[['yhat', 'yhat_lower', 'yhat_upper']] = forecast[['yhat', 'yhat_lower', 'yhat_upper']].clip(lower=0)

                self.all_predictions[vagone] = {
                    'forecast': forecast,
                    'avg_historical': df_prophet['y'].mean(),
                    'avg_forecast': forecast['yhat'].tail(self.forecast_days).mean()
                }
                logger.info(f"✅ Model created for {vagone}")

            except Exception as e:
                logger.error(f"❌ Error for {vagone}: {str(e)}")
                continue

        logger.info(f"Completed! Models created for {len(self.all_predictions)} trains")
    
    def create_summary_visualizations(self):
        """Create the summary visualizations with two heatmaps and save to file."""
        logger.info("Creating summary visualizations (confronto e media)...")

        vagoni = list(self.all_predictions.keys())

        # 1. FORECAST DATA - Average by day of the week
        forecast_data = {}
        for vagone, data in self.all_predictions.items():
            future_forecast = data['forecast']
            last_date = self.df_filtered[self.df_filtered['Name_mode'] == vagone]['ts'].max()
            future_only = future_forecast[future_forecast['ds'] > last_date].head(30)
            future_only['day_of_week'] = future_only['ds'].dt.day_name()
            weekly_forecast = future_only.groupby('day_of_week')['yhat'].mean()
            forecast_data[vagone] = weekly_forecast
            
        forecast_weekly = pd.DataFrame(forecast_data).T
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        forecast_weekly = forecast_weekly.reindex(columns=[d for d in days_order if d in forecast_weekly.columns])

        # 2. COMPARISON DATA: Historical vs Forecasts
        comparison_data = {}
        for vagone in vagoni:
            hist_avg = self.all_predictions[vagone]['avg_historical']
            pred_avg = self.all_predictions[vagone]['avg_forecast']
            comparison_data[vagone] = {
                'Storico': hist_avg, 'Previsioni': pred_avg, 'Differenza': pred_avg - hist_avg,
                'Variazione_%': ((pred_avg - hist_avg) / hist_avg * 100) if hist_avg > 0 else 0
            }
        comparison_df = pd.DataFrame(comparison_data).T

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
        fig.suptitle('Analisi Predittiva Allarmi Ferroviari', fontsize=18, fontweight='bold')

        # Heatmap 1: Forecasts by day of week ("media")
        sns.heatmap(forecast_weekly, annot=True, fmt='.1f', cmap='Blues', ax=ax1, cbar_kws={'label': 'Allarmi Previsti'})
        ax1.set_title('📊 PREVISIONI: Media per Giorno Settimana (prossimi 30gg)', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Giorno della Settimana')
        ax1.set_ylabel('Vagone')

        # Heatmap 2: Comparison Historical vs Forecasts ("confronto")
        comparison_subset = comparison_df[['Storico', 'Previsioni', 'Differenza']]
        sns.heatmap(comparison_subset.T, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax2, cbar_kws={'label': 'Allarmi'})
        ax2.set_title('🔄 CONFRONTO: Media Storica vs Media Prevista', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Vagone')
        ax2.set_ylabel('Metrica')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save figure
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = 'railway_summary_predictions.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Summary visualizations saved to {filename}")
        
        # <<< MODIFICATION: Close the plot to free memory and prevent display blocking
        plt.close(fig)
        
        self._generate_summary_report(comparison_df, forecast_weekly)

    def create_daily_forecast_heatmap(self):
        """Create a heatmap showing the daily forecast for the next 7 days and save to file."""
        logger.info("Creating 7-day daily forecast visualization...")

        daily_data = {}
        for vagone, data in self.all_predictions.items():
            forecast_df = data['forecast']
            last_historical_date = self.df_filtered[self.df_filtered['Name_mode'] == vagone]['ts'].max()
            future_only = forecast_df[forecast_df['ds'] > last_historical_date].head(7)
            daily_series = pd.Series(future_only['yhat'].values, index=future_only['ds'])
            daily_data[vagone] = daily_series

        daily_forecast_df = pd.DataFrame(daily_data).T
        if daily_forecast_df.empty:
            logger.warning("Could not generate 7-day forecast heatmap, no data available.")
            return
            
        logger.info(f"7-Day forecast data shape: {daily_forecast_df.shape}")

        daily_forecast_df.columns = [d.strftime('%Y-%m-%d (%a)') for d in daily_forecast_df.columns]

        # Create the heatmap in a new figure
        fig, ax = plt.subplots(figsize=(18, 10))
        sns.heatmap(
            daily_forecast_df,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            linewidths=.5,
            cbar_kws={'label': 'Numero Previsto di Allarmi Critici'},
            ax=ax
        )
        
        ax.set_title('🗓️ PREVISIONI GIORNALIERE: Prossimi 7 Giorni', fontsize=16, fontweight='bold')
        ax.set_xlabel('Data della Previsione', fontsize=12)
        ax.set_ylabel('Vagone', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = 'railway_7day_forecast.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"7-day forecast visualization saved to {filename}")
        
        # <<< MODIFICATION: Close the plot to free memory and prevent display blocking
        plt.close(fig)

    def create_individual_train_plots(self):
        """Create individual Prophet-style plots for each train"""
        logger.info("Creating individual train forecast plots...")
        
        # Create directory for individual plots
        plots_dir = 'train_plots'
        os.makedirs(plots_dir, exist_ok=True)
        
        for vagone, prediction_data in self.all_predictions.items():
            try:
                # Get historical data
                historical_data = self.df_filtered[self.df_filtered['Name_mode'] == vagone].copy()
                forecast_df = prediction_data['forecast']
                
                # Split historical and future data
                last_historical_date = historical_data['ts'].max()
                historical_forecast = forecast_df[forecast_df['ds'] <= last_historical_date]
                future_forecast = forecast_df[forecast_df['ds'] > last_historical_date]
                
                # Create the plot
                fig, ax = plt.subplots(figsize=(15, 8))
                
                # Plot historical actual data
                ax.plot(historical_data['ts'], historical_data['Criticality_1'], 
                       'o-', color='blue', linewidth=2, markersize=4, label=f'Training Data ({len(historical_data)} points)', alpha=0.8)
                
                # Plot historical forecast (fitted values) - optional, for model validation
                ax.plot(historical_forecast['ds'], historical_forecast['yhat'], 
                       '--', color='orange', linewidth=1.5, alpha=0.7, label='Historical Fit')
                
                # Plot future predictions
                ax.plot(future_forecast['ds'], future_forecast['yhat'], 
                       'o-', color='red', linewidth=2.5, markersize=5, label=f'Previsioni su Test ({len(future_forecast)} giorni)')
                
                # Plot confidence intervals for future predictions
                ax.fill_between(future_forecast['ds'], 
                              future_forecast['yhat_lower'], 
                              future_forecast['yhat_upper'],
                              color='red', alpha=0.2, label='Confidenza 95%')
                
                # Add vertical line to separate historical from predictions
                ax.axvline(x=last_historical_date, color='gray', linestyle=':', alpha=0.7, linewidth=2)
                ax.text(last_historical_date, ax.get_ylim()[1]*0.9, 'Inizio Previsioni', 
                       rotation=90, verticalalignment='top', fontsize=10, alpha=0.7)
                
                # Calculate and display key metrics
                avg_historical = prediction_data['avg_historical']
                avg_forecast = prediction_data['avg_forecast']
                change_pct = ((avg_forecast - avg_historical) / avg_historical * 100) if avg_historical > 0 else 0
                
                # Add performance metrics if available
                metrics_text = f"Media Storica: {avg_historical:.1f}\nMedia Prevista: {avg_forecast:.1f}\nVariazione: {change_pct:+.1f}%"
                if vagone in self.performance_metrics:
                    perf = self.performance_metrics[vagone]
                    metrics_text += f"\n\nQualità Modello:\nMAE: {perf['mae']:.2f}\nMAPE: {perf['mape']:.1f}%"
                
                # Add text box with metrics
                ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                # Formatting
                ax.set_title(f'🚆 Analisi Predittiva Allarmi - {vagone}', fontsize=16, fontweight='bold', pad=20)
                ax.set_xlabel('Data', fontsize=12)
                ax.set_ylabel('Numero Allarmi Critici', fontsize=12)
                ax.legend(loc='upper right', framealpha=0.9)
                ax.grid(True, alpha=0.3)
                
                # Format x-axis dates
                fig.autofmt_xdate()
                
                # Set y-axis to start from 0
                ax.set_ylim(bottom=0)
                
                plt.tight_layout()
                
                # Save individual plot
                plot_filename = os.path.join(plots_dir, f'forecast_{vagone.replace("/", "_")}.png')
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                logger.info(f"✅ Plot saved for {vagone}")
                
            except Exception as e:
                logger.error(f"❌ Error creating plot for {vagone}: {str(e)}")
                continue
        
        logger.info(f"📊 Individual plots saved in directory: {plots_dir}")
        return plots_dir

    def create_analysis_json(self):
        """Create JSON with historical data, forecasts, differences and performance metrics"""
        logger.info("Creating comprehensive analysis JSON...")
        
        analysis_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'forecast_days': self.forecast_days,
                'total_trains': len(self.all_predictions),
                'data_period': {
                    'start': self.df_filtered['ts'].min().isoformat(),
                    'end': self.df_filtered['ts'].max().isoformat()
                }
            },
            'performance_metrics': self.performance_metrics,
            'train_analysis': {}
        }
        
        # Analysis for each train
        for vagone, prediction_data in self.all_predictions.items():
            # Historical statistics
            historical_data = self.df_filtered[self.df_filtered['Name_mode'] == vagone]['Criticality_1']
            
            # Future forecasts
            forecast_df = prediction_data['forecast']
            last_date = self.df_filtered[self.df_filtered['Name_mode'] == vagone]['ts'].max()
            future_forecast = forecast_df[forecast_df['ds'] > last_date].head(self.forecast_days)
            
            # Calculate comparison metrics
            avg_historical = float(prediction_data['avg_historical'])
            avg_forecast = float(prediction_data['avg_forecast'])
            difference = avg_forecast - avg_historical
            percentage_change = (difference / avg_historical * 100) if avg_historical > 0 else 0
            
            analysis_data['train_analysis'][vagone] = {
                'storico': {
                    'media': round(avg_historical, 2),
                    'totale': int(historical_data.sum()),
                    'massimo': int(historical_data.max()),
                    'minimo': int(historical_data.min()),
                    'deviazione_standard': round(float(historical_data.std()), 2)
                },
                'previsione': {
                    'media': round(avg_forecast, 2),
                    'totale_previsto': round(float(future_forecast['yhat'].sum()), 1),
                    'massimo_previsto': round(float(future_forecast['yhat'].max()), 2),
                    'minimo_previsto': round(float(future_forecast['yhat'].min()), 2)
                },
                'confronto': {
                    'differenza_assoluta': round(difference, 2),
                    'variazione_percentuale': round(percentage_change, 1),
                    'tendenza': 'aumento' if difference > 0 else 'diminuzione' if difference < 0 else 'stabile'
                },
                'previsioni_giornaliere': [
                    {
                        'data': row['ds'].strftime('%Y-%m-%d'),
                        'allarmi_previsti': round(float(row['yhat']), 2),
                        'limite_inferiore': round(float(row['yhat_lower']), 2),
                        'limite_superiore': round(float(row['yhat_upper']), 2)
                    }
                    for _, row in future_forecast.iterrows()
                ]
            }
        
        # Save JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_filename = 'railway_analysis.json'
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis JSON saved to {json_filename}")
        return json_filename
        """Create JSON with historical data, forecasts, differences and performance metrics"""
        logger.info("Creating comprehensive analysis JSON...")
        
        analysis_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'forecast_days': self.forecast_days,
                'total_trains': len(self.all_predictions),
                'data_period': {
                    'start': self.df_filtered['ts'].min().isoformat(),
                    'end': self.df_filtered['ts'].max().isoformat()
                }
            },
            'performance_metrics': self.performance_metrics,
            'train_analysis': {}
        }
        
        # Analysis for each train
        for vagone, prediction_data in self.all_predictions.items():
            # Historical statistics
            historical_data = self.df_filtered[self.df_filtered['Name_mode'] == vagone]['Criticality_1']
            
            # Future forecasts
            forecast_df = prediction_data['forecast']
            last_date = self.df_filtered[self.df_filtered['Name_mode'] == vagone]['ts'].max()
            future_forecast = forecast_df[forecast_df['ds'] > last_date].head(self.forecast_days)
            
            # Calculate comparison metrics
            avg_historical = float(prediction_data['avg_historical'])
            avg_forecast = float(prediction_data['avg_forecast'])
            difference = avg_forecast - avg_historical
            percentage_change = (difference / avg_historical * 100) if avg_historical > 0 else 0
            
            analysis_data['train_analysis'][vagone] = {
                'storico': {
                    'media': round(avg_historical, 2),
                    'totale': int(historical_data.sum()),
                    'massimo': int(historical_data.max()),
                    'minimo': int(historical_data.min()),
                    'deviazione_standard': round(float(historical_data.std()), 2)
                },
                'previsione': {
                    'media': round(avg_forecast, 2),
                    'totale_previsto': round(float(future_forecast['yhat'].sum()), 1),
                    'massimo_previsto': round(float(future_forecast['yhat'].max()), 2),
                    'minimo_previsto': round(float(future_forecast['yhat'].min()), 2)
                },
                'confronto': {
                    'differenza_assoluta': round(difference, 2),
                    'variazione_percentuale': round(percentage_change, 1),
                    'tendenza': 'aumento' if difference > 0 else 'diminuzione' if difference < 0 else 'stabile'
                },
                'previsioni_giornaliere': [
                    {
                        'data': row['ds'].strftime('%Y-%m-%d'),
                        'allarmi_previsti': round(float(row['yhat']), 2),
                        'limite_inferiore': round(float(row['yhat_lower']), 2),
                        'limite_superiore': round(float(row['yhat_upper']), 2)
                    }
                    for _, row in future_forecast.iterrows()
                ]
            }
        
        # Save JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_filename = 'railway_analysis.json'
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis JSON saved to {json_filename}")
        return json_filename

    def _generate_summary_report(self, comparison_df, forecast_weekly):
        """Generate a summary report to the logger"""
        logger.info("\n" + "="*60 + "\n📊 REPORT RIEPILOGATIVO PREVISIONI\n" + "="*60)
        logger.info("\n🗓️ GIORNI PIÙ CRITICI PREVISTI (media su tutti i vagoni):")
        worst_days = forecast_weekly.mean().sort_values(ascending=False)
        for day, value in worst_days.head(3).items():
            logger.info(f"   • {day}: {value:.1f} allarmi medi previsti")

        logger.info("\n⚠️ VAGONI A RISCHIO (previsioni > storico):")
        at_risk = comparison_df[comparison_df['Variazione_%'] > 20].sort_values('Variazione_%', ascending=False)
        if not at_risk.empty:
            for train, row in at_risk.iterrows():
                logger.info(f"   • {train}: +{row['Variazione_%']:.1f}% ({row['Storico']:.1f} → {row['Previsioni']:.1f})")
        else:
            logger.info("   Nessun vagone con aumento significativo previsto.")
            
        logger.info("\n✅ VAGONI IN MIGLIORAMENTO (previsioni < storico):")
        improving = comparison_df[comparison_df['Variazione_%'] < -20].sort_values('Variazione_%')
        if not improving.empty:
            for train, row in improving.iterrows():
                logger.info(f"   • {train}: {row['Variazione_%']:.1f}% ({row['Storico']:.1f} → {row['Previsioni']:.1f})")
        else:
            logger.info("   Nessun vagone con miglioramento significativo previsto.")
            
        # Performance metrics summary
        if self.performance_metrics:
            logger.info("\n📈 QUALITÀ MODELLI:")
            avg_mae = np.mean([m['mae'] for m in self.performance_metrics.values()])
            avg_mape = np.mean([m['mape'] for m in self.performance_metrics.values()])
            logger.info(f"   • MAE medio: {avg_mae:.2f}")
            logger.info(f"   • MAPE medio: {avg_mape:.1f}%")
            logger.info(f"   • Modelli valutati: {len(self.performance_metrics)}")
            
        report_filename = 'railway_report.csv'
        comparison_df.to_csv(report_filename)
        logger.info(f"\n💾 Report dettagliato salvato in: {report_filename}")

    def run(self):
        """Main execution method"""
        try:
            logger.info("🚆 Starting Railway Alarm Prediction System")
            self.load_and_preprocess_data()
            self.predict_all_trains()

            if self.all_predictions:
                self.create_summary_visualizations()
                self.create_daily_forecast_heatmap()
                
                # Create individual train plots
                plots_dir = self.create_individual_train_plots()
                logger.info(f"📈 Individual train plots saved in: {plots_dir}")
                
                json_file = self.create_analysis_json()
                logger.info(f"📋 Comprehensive JSON analysis: {json_file}")
            else:
                logger.error("No predictions generated. Check your data.")

            logger.info("✅ Process completed successfully!")

        except Exception as e:
            logger.error(f"❌ Fatal error: {str(e)}")
            raise


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Railway Alarm Prediction System')
    parser.add_argument('--file', '-f', type=str, default='data/TSR_040_daily.csv', help='Path to the input CSV file')
    parser.add_argument('--days', '-d', type=int, default=30, help='Number of days to forecast')
    parser.add_argument('--output-dir', '-o', type=str, default='.', help='Output directory for results')
    
    args = parser.parse_args()
    
    if args.output_dir != '.':
        os.makedirs(args.output_dir, exist_ok=True)
        os.chdir(args.output_dir)
    
    predictor = RailwayAlarmPredictor(file_path=args.file, forecast_days=args.days)
    predictor.run()


if __name__ == "__main__":
    main()