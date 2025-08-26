#!/usr/bin/env python3
"""
Model Improvement Analysis Script

This script replicates the functionality of model_improvement.ipynb.
It compares model predictions vs actual values by processing MLflow results
and raw metrics data to generate comparison plots.

Usage:
    python model_improvement_analysis.py <df_results_path> <df_metrics_path> <output_image_path> [--df_results_path_2 PATH] [--model1_label LABEL] [--model2_label LABEL] [--split_12h]

Arguments:
    df_results_path: Path to the MLflow results CSV file or folder
    df_metrics_path: Path to the raw metrics CSV file or folder  
    output_image_path: Path where to save the comparison plot
    --df_results_path_2: Optional second model results CSV or folder; if provided, its predictions are added to the plots
    --model1_label: Legend label for first model (default: 'Predicted')
    --model2_label: Legend label for second model (default: 'Model 2')
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import os
import sys
from typing import Optional, List


def _load_results_df(df_results_path: str) -> pd.DataFrame:
    """Load MLflow results from a file or a folder of CSVs (with ';' sep).

    When a folder is given, only files containing '_mlflow_results' are read.
    """
    if os.path.isdir(df_results_path):
        print(f"Loading multiple CSV files from folder: {df_results_path}")
        csv_files = [f for f in os.listdir(df_results_path)
                    if f.endswith('.csv') and '_mlflow_results' in f]
        
        if not csv_files:
            raise ValueError(f"No MLflow results CSV files found in directory: {df_results_path}")
        
        print(f"Found {len(csv_files)} MLflow results CSV files")
        
        # Read and combine all CSV files
        dfs: List[pd.DataFrame] = []
        for filename in csv_files:
            filepath = os.path.join(df_results_path, filename)
            print(f"  Loading: {filename}")
            df = pd.read_csv(filepath, sep=';')
            dfs.append(df)
        
        # Concatenate all dataframes
        df_results = pd.concat(dfs, ignore_index=True)
        print(f"Combined {len(dfs)} MLflow results files into single dataframe")
    else:
        # Single file
        print(f"Loading single CSV file: {df_results_path}")
        df_results = pd.read_csv(df_results_path, sep=';')

    # Clean results dataframe
    df_results = df_results.drop(columns=["cluster_1", "qos_1"], errors='ignore')
    return df_results


def load_and_prepare_data(df_results_path, df_metrics_path):
    """
    Load and prepare the results and metrics dataframes.
    
    Parameters:
    - df_results_path: Path to MLflow results CSV file OR folder containing multiple CSV files
    - df_metrics_path: Path to raw metrics CSV
    
    Returns:
    - df_results: Prepared results dataframe
    - df_metrics: Prepared metrics dataframe
    """
    print("Loading data...")
    
    # Load results (file or folder)
    df_results = _load_results_df(df_results_path)
    
    # Load metrics
    if os.path.isdir(df_metrics_path):
        print(f"Loading multiple CSV files from metrics folder: {df_metrics_path}")
        csv_files = [f for f in os.listdir(df_metrics_path) if f.endswith('.csv')]
        
        if not csv_files:
            raise ValueError(f"No CSV files found in directory: {df_metrics_path}")
        
        print(f"Found {len(csv_files)} metrics CSV files")
        
        # Read and combine all CSV files
        dfs = []
        for filename in csv_files:
            filepath = os.path.join(df_metrics_path, filename)
            print(f"  Loading: {filename}")
            df = pd.read_csv(filepath)
            dfs.append(df)
        
        # Concatenate all dataframes and sort by date
        df_metrics = pd.concat(dfs, ignore_index=True).sort_values(by='date', na_position='last')
        print(f"Combined {len(dfs)} metrics files into single dataframe")
    else:
        # Single metrics file
        print(f"Loading single metrics CSV file: {df_metrics_path}")
        df_metrics = pd.read_csv(df_metrics_path)
    
    print(f"Results shape: {df_results.shape}")
    print(f"Metrics shape: {df_metrics.shape}")
    
    return df_results, df_metrics


def add_previous_metrics(df_results, df_metrics):
    """
    Add previous mean and max latency for the 5-minute window before each prediction.
    
    Parameters:
    - df_results: MLflow results with avg_qos_0 and max_qos_0 predictions
    - df_metrics: Raw metrics with pipelines_status_realtime_pipeline_latency
    
    Returns:
    - df_results with previous_mean and previous_max columns added
    """
    print("Adding previous metrics...")
    
    df_results_copy = df_results.copy()
    
    # Convert timestamps
    df_results_copy['start_time'] = pd.to_datetime(df_results_copy['start_time'], unit='s')
    df_metrics_copy = df_metrics.copy()
    df_metrics_copy['date'] = pd.to_datetime(df_metrics_copy['date'])
    
    # Initialize columns
    df_results_copy['previous_mean'] = None
    df_results_copy['previous_max'] = None
    
    # For each prediction, get the 5 minutes of data before it
    for idx, row in df_results_copy.iterrows():
        prediction_time = row['start_time'].replace(second=0, microsecond=0)
        window_start = prediction_time - timedelta(minutes=5)
        window_end = prediction_time
        
        # Get data from this 5-minute window
        window_data = df_metrics_copy[
            (df_metrics_copy['date'] >= window_start) & 
            (df_metrics_copy['date'] < window_end)
        ]
        
        if not window_data.empty:
            latency_values = window_data['pipelines_status_realtime_pipeline_latency'].dropna()
            
            if not latency_values.empty:
                df_results_copy.loc[idx, 'previous_mean'] = latency_values.mean()
                df_results_copy.loc[idx, 'previous_max'] = latency_values.max()
    
    return df_results_copy


def print_data_summary(df_results_final):
    """
    Print summary of the combined data to verify functionality.
    
    Parameters:
    - df_results_final: Final dataframe with previous and predicted values
    """
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    print(f"Total samples: {len(df_results_final)}")
    print(f"Columns: {list(df_results_final.columns)}")
    
    # Check for key columns
    key_columns = ['start_time', 'previous_mean', 'previous_max', 'avg_qos_0', 'max_qos_0']
    missing_cols = [col for col in key_columns if col not in df_results_final.columns]
    if missing_cols:
        print(f"Missing expected columns: {missing_cols}")
    
    print(f"\nFirst 5 rows:")
    print(df_results_final.head())
    
    # Check data availability
    print(f"\nData availability:")
    for col in ['previous_mean', 'previous_max']:
        if col in df_results_final.columns:
            non_null = df_results_final[col].notna().sum()
            print(f"  {col}: {non_null}/{len(df_results_final)} ({non_null/len(df_results_final)*100:.1f}%)")
    
    for col in ['avg_qos_0', 'max_qos_0']:
        if col in df_results_final.columns:
            non_null = df_results_final[col].notna().sum()
            print(f"  {col}: {non_null}/{len(df_results_final)} ({non_null/len(df_results_final)*100:.1f}%)")


def clean_dataframe(df_results_final):
    """
    Clean the dataframe by removing unnecessary MLflow columns.
    
    Parameters:
    - df_results_final: Final results dataframe
    
    Returns:
    - Cleaned dataframe
    """
    columns_to_drop = ['mlflow.user', 'mlflow.source.name', 
                      'mlflow.source.type', 'mlflow.runName']
    
    return df_results_final.drop(columns=columns_to_drop, errors='ignore')


def plot_predictions_vs_previous(df_results_final, save_path, split_12h=False):
    """
    Plot mean and max values comparing predictions vs previous values.
    
    Parameters:
    - df_results_final: DataFrame with previous and predicted values
    - save_path: Path to save the image(s)
    - split_12h: If True, split into 12-hour windows and create multiple plots
    """
    # Ensure start_time is datetime
    df_results_final['start_time'] = pd.to_datetime(df_results_final['start_time'])
    
    if not split_12h:
        # Single plot for all data
        _create_single_plot(df_results_final, save_path)
    else:
        # Split into 12-hour windows
        _create_12h_plots(df_results_final, save_path)


def _create_single_plot(df_data, save_path, model1_label: str = 'Predicted', model2_label: Optional[str] = None):
    """Create a single plot with all data."""
    print(f"Creating prediction comparison plot...")
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Mean values
    ax1.plot(df_data['start_time'],
             df_data['previous_mean'],
             label='Previous Mean', linewidth=2, color='blue', marker='o', markersize=3)
    ax1.plot(df_data['start_time'],
             df_data['avg_qos_0'],
             label=f'{model1_label} Mean', linewidth=2, color='red', marker='s', markersize=3)
    if 'avg_qos_0_model2' in df_data.columns:
        ax1.plot(df_data['start_time'],
                 df_data['avg_qos_0_model2'],
                 label=f'{model2_label or "Model 2"} Mean', linewidth=2, color='green', marker='^', markersize=3)
    
    ax1.set_title('Mean Pipeline Latency: Previous vs Predicted', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Mean Latency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Max values
    ax2.plot(df_data['start_time'],
             df_data['previous_max'],
             label='Previous Max', linewidth=2, color='blue', marker='o', markersize=3)
    ax2.plot(df_data['start_time'],
             df_data['max_qos_0'],
             label=f'{model1_label} Max', linewidth=2, color='red', marker='s', markersize=3)
    if 'max_qos_0_model2' in df_data.columns:
        ax2.plot(df_data['start_time'],
                 df_data['max_qos_0_model2'],
                 label=f'{model2_label or "Model 2"} Max', linewidth=2, color='green', marker='^', markersize=3)
    
    ax2.set_title('Max Pipeline Latency: Previous vs Predicted', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Max Latency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    
    plt.close()


def _create_12h_plots(df_data, save_path, model1_label: str = 'Predicted', model2_label: Optional[str] = None):
    """Create multiple plots split by 12-hour windows."""
    print(f"Creating 12-hour split plots...")
    
    # Sort by time
    df_data = df_data.sort_values('start_time').reset_index(drop=True)
    
    # Get time range
    start_time = df_data['start_time'].min()
    end_time = df_data['start_time'].max()
    
    # Create 12-hour windows
    current_time = start_time
    plot_count = 0
    
    while current_time < end_time:
        window_end = current_time + pd.Timedelta(hours=12)
        
        # Filter data for this 12-hour window
        window_data = df_data[
            (df_data['start_time'] >= current_time) & 
            (df_data['start_time'] < window_end)
        ]
        
        if not window_data.empty:
            plot_count += 1
            
            # Generate filename for this window
            base_path = os.path.splitext(save_path)[0]
            extension = os.path.splitext(save_path)[1] or '.png'
            window_filename = f"{base_path}_12h_window_{plot_count:02d}_{current_time.strftime('%Y%m%d_%H%M')}_to_{window_end.strftime('%Y%m%d_%H%M')}{extension}"
            
            # Create plot for this window
            _create_window_plot(window_data, window_filename, current_time, window_end, plot_count, model1_label, model2_label)
        
        current_time = window_end
    
    print(f"Created {plot_count} 12-hour window plots")


def _create_window_plot(df_data, save_path, window_start, window_end, window_num, model1_label: str = 'Predicted', model2_label: Optional[str] = None):
    """Create a plot for a specific 12-hour window."""
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Mean values
    ax1.plot(df_data['start_time'],
             df_data['previous_mean'],
             label='Previous Mean', linewidth=2, color='blue', marker='o', markersize=4)
    ax1.plot(df_data['start_time'],
             df_data['avg_qos_0'],
             label=f'{model1_label} Mean', linewidth=2, color='red', marker='s', markersize=4)
    if 'avg_qos_0_model2' in df_data.columns:
        ax1.plot(df_data['start_time'],
                 df_data['avg_qos_0_model2'],
                 label=f'{model2_label or "Model 2"} Mean', linewidth=2, color='green', marker='^', markersize=4)
    
    ax1.set_title(f'Mean Pipeline Latency: Previous vs Predicted (Window {window_num}: {window_start.strftime("%Y-%m-%d %H:%M")} - {window_end.strftime("%Y-%m-%d %H:%M")})', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Mean Latency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Max values
    ax2.plot(df_data['start_time'],
             df_data['previous_max'],
             label='Previous Max', linewidth=2, color='blue', marker='o', markersize=4)
    ax2.plot(df_data['start_time'],
             df_data['max_qos_0'],
             label=f'{model1_label} Max', linewidth=2, color='red', marker='s', markersize=4)
    if 'max_qos_0_model2' in df_data.columns:
        ax2.plot(df_data['start_time'],
                 df_data['max_qos_0_model2'],
                 label=f'{model2_label or "Model 2"} Max', linewidth=2, color='green', marker='^', markersize=4)
    
    ax2.set_title(f'Max Pipeline Latency: Previous vs Predicted (Window {window_num})', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Max Latency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Window {window_num} plot saved to: {save_path}")
    
    plt.close()


def print_summary_statistics(df_results_final):
    """
    Print summary statistics for the analysis.
    
    Parameters:
    - df_results_final: Final results dataframe
    """
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"Total samples: {len(df_results_final)}")
    print(f"Date range: {df_results_final['start_time'].min()} to {df_results_final['start_time'].max()}")
    
    # Real values statistics
    real_mean = df_results_final['real_future_values_5_min_mean'].dropna()
    real_max = df_results_final['real_future_values_5_min_max'].dropna()
    
    print(f"\nReal Mean Latency:")
    print(f"  Count: {len(real_mean)}")
    print(f"  Mean: {real_mean.mean():.6f}")
    print(f"  Std: {real_mean.std():.6f}")
    print(f"  Min: {real_mean.min():.6f}")
    print(f"  Max: {real_mean.max():.6f}")
    
    print(f"\nReal Max Latency:")
    print(f"  Count: {len(real_max)}")
    print(f"  Mean: {real_max.mean():.6f}")
    print(f"  Std: {real_max.std():.6f}")
    print(f"  Min: {real_max.min():.6f}")
    print(f"  Max: {real_max.max():.6f}")
    
    # QoS statistics
    qos = df_results_final['qos_0'].dropna()
    print(f"\nQoS Values:")
    print(f"  Count: {len(qos)}")
    print(f"  Mean: {qos.mean():.6f}")
    print(f"  Std: {qos.std():.6f}")
    print(f"  Min: {qos.min():.6f}")
    print(f"  Max: {qos.max():.6f}")
    
    # Predicted statistics (if available)
    pred_mean = df_results_final['mean_predicted'].dropna()
    pred_max = df_results_final['max_predicted'].dropna()
    
    if len(pred_mean) > 0:
        print(f"\nPredicted Mean Latency:")
        print(f"  Count: {len(pred_mean)}")
        print(f"  Mean: {pred_mean.mean():.6f}")
        print(f"  Std: {pred_mean.std():.6f}")
        print(f"  Min: {pred_mean.min():.6f}")
        print(f"  Max: {pred_mean.max():.6f}")
    else:
        print(f"\nPredicted Mean Latency: No data available")
    
    if len(pred_max) > 0:
        print(f"\nPredicted Max Latency:")
        print(f"  Count: {len(pred_max)}")
        print(f"  Mean: {pred_max.mean():.6f}")
        print(f"  Std: {pred_max.std():.6f}")
        print(f"  Min: {pred_max.min():.6f}")
        print(f"  Max: {pred_max.max():.6f}")
    else:
        print(f"\nPredicted Max Latency: No data available")

    # Model 2 statistics when present
    if 'avg_qos_0_model2' in df_results_final.columns:
        model2_mean = df_results_final['avg_qos_0_model2'].dropna()
        model2_max = df_results_final['max_qos_0_model2'].dropna()
        print(f"\nModel 2 Predicted Mean Latency:")
        if len(model2_mean) > 0:
            print(f"  Count: {len(model2_mean)}")
            print(f"  Mean: {model2_mean.mean():.6f}")
            print(f"  Std: {model2_mean.std():.6f}")
            print(f"  Min: {model2_mean.min():.6f}")
            print(f"  Max: {model2_mean.max():.6f}")
        else:
            print("  No data available")
        print(f"\nModel 2 Predicted Max Latency:")
        if len(model2_max) > 0:
            print(f"  Count: {len(model2_max)}")
            print(f"  Mean: {model2_max.mean():.6f}")
            print(f"  Std: {model2_max.std():.6f}")
            print(f"  Min: {model2_max.min():.6f}")
            print(f"  Max: {model2_max.max():.6f}")
        else:
            print("  No data available")


def merge_second_model_predictions(df_results_final: pd.DataFrame, df_results_model2: pd.DataFrame) -> pd.DataFrame:
    """Merge predictions from a second model into the main dataframe.

    Aligns by minute-resolution timestamps. Expects the second dataframe to have
    'start_time' (epoch seconds) and prediction columns 'avg_qos_0' and 'max_qos_0'.
    The merged columns are named 'avg_qos_0_model2' and 'max_qos_0_model2'.
    """
    df_left = df_results_final.copy()
    df_right = df_results_model2.copy()

    # Ensure datetime and minute alignment keys exist
    if not pd.api.types.is_datetime64_any_dtype(df_left['start_time']):
        df_left['start_time'] = pd.to_datetime(df_left['start_time'], unit='s')
    df_left['start_time_minute'] = df_left['start_time'].dt.floor('T')

    if not pd.api.types.is_datetime64_any_dtype(df_right['start_time']):
        df_right['start_time'] = pd.to_datetime(df_right['start_time'], unit='s')
    df_right['start_time_minute'] = df_right['start_time'].dt.floor('T')

    # Select and rename model2 prediction columns
    cols_available = set(df_right.columns)
    needed = {'avg_qos_0', 'max_qos_0'}
    if not needed.issubset(cols_available):
        missing = list(needed - cols_available)
        raise ValueError(f"Second results missing expected columns: {missing}")

    df_right_small = df_right[['start_time_minute', 'avg_qos_0', 'max_qos_0']].copy()
    df_right_small = df_right_small.rename(columns={
        'avg_qos_0': 'avg_qos_0_model2',
        'max_qos_0': 'max_qos_0_model2',
    })

    merged = df_left.merge(df_right_small, on='start_time_minute', how='left')
    return merged


def _read_csv_auto_sep(filepath: str) -> pd.DataFrame:
    """Read CSV trying ';' then ',' separators to handle different sources."""
    try:
        df = pd.read_csv(filepath, sep=';')
        # Heuristic: if only 1 column or header contains commas, retry
        if df.shape[1] == 1 or (len(df.columns) == 1 and ',' in df.columns[0]):
            raise ValueError('Bad sep ; assumed')
        return df
    except Exception:
        return pd.read_csv(filepath, sep=',')


def _load_model2_input(path: str) -> tuple[pd.DataFrame, str]:
    """Load second model input which can be either:
    - results-style with 'start_time', 'avg_qos_0', 'max_qos_0'
    - metrics-style with 'date', 'pipelines_status_realtime_pipeline_latency'

    Returns: (dataframe, kind) where kind in {'results', 'metrics'}
    """
    if os.path.isdir(path):
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        if not csv_files:
            raise ValueError(f"No CSV files found in directory: {path}")
        dfs: List[pd.DataFrame] = []
        for filename in csv_files:
            df = _read_csv_auto_sep(os.path.join(path, filename))
            dfs.append(df)
        df_all = pd.concat(dfs, ignore_index=True)
    else:
        df_all = _read_csv_auto_sep(path)

    cols = set(df_all.columns)
    if {'start_time', 'avg_qos_0', 'max_qos_0'}.issubset(cols):
        return df_all, 'results'
    if {'date', 'pipelines_status_realtime_pipeline_latency'}.issubset(cols):
        return df_all, 'metrics'
    raise ValueError("Unsupported second model file schema. Expect either MLflow-style results with ['start_time','avg_qos_0','max_qos_0'] or metrics with ['date','pipelines_status_realtime_pipeline_latency'].")


def add_model2_from_metrics(df_results_final: pd.DataFrame, df_model2_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute model2 5-minute mean and max AFTER each MLflow prediction timestamp.

    Window is future: (start_time, start_time + 5 minutes], matching 10 steps at 30s cadence
    and plotted at the MLflow 'start_time'.

    - df_results_final must contain 'start_time' (datetime or epoch seconds)
    - df_model2_metrics must contain 'date' (ISO or datetime) and 'pipelines_status_realtime_pipeline_latency'
    Adds columns: 'avg_qos_0_model2' and 'max_qos_0_model2'.
    """
    df_out = df_results_final.copy()
    # Ensure datetime types
    if not pd.api.types.is_datetime64_any_dtype(df_out['start_time']):
        df_out['start_time'] = pd.to_datetime(df_out['start_time'], unit='s')
    df_model2 = df_model2_metrics.copy()
    df_model2['date'] = pd.to_datetime(df_model2['date'])

    # Initialize columns
    df_out['avg_qos_0_model2'] = None
    df_out['max_qos_0_model2'] = None

    value_col = 'pipelines_status_realtime_pipeline_latency'
    if value_col not in df_model2.columns:
        raise ValueError(f"Model2 metrics missing column: {value_col}")

    for idx, row in df_out.iterrows():
        prediction_time = row['start_time'].replace(second=0, microsecond=0)
        window_start = prediction_time
        window_end = prediction_time + timedelta(minutes=5)

        # Future window: (start_time, start_time + 5min]
        window_data = df_model2[(df_model2['date'] > window_start) & (df_model2['date'] <= window_end)][value_col].dropna()
        if not window_data.empty:
            df_out.loc[idx, 'avg_qos_0_model2'] = window_data.mean()
            df_out.loc[idx, 'max_qos_0_model2'] = window_data.max()

    return df_out


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(
        description='Model Improvement Analysis - Compare predictions vs actual values',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python proactive_migration_validation_analysis.py results.csv metrics.csv output.png
    python proactive_migration_validation_analysis.py /path/to/results_folder/ /path/to/metrics.csv /path/to/output.png
    python proactive_migration_validation_analysis.py /path/to/results.csv /path/to/metrics.csv /path/to/output.png
        """
    )
    
    parser.add_argument('df_results_path', 
                       help='Path to the MLflow results CSV file or folder containing multiple CSV files')
    parser.add_argument('df_metrics_path', 
                       help='Path to the raw metrics CSV file or folder containing multiple CSV files')
    parser.add_argument('output_image_path', 
                       help='Path where to save the comparison plot')
    parser.add_argument('--df_results_path_2', default=None,
                       help="Optional: path to another model's MLflow results CSV or folder")
    parser.add_argument('--model1_label', default='Predicted',
                       help="Legend label for the first model's predictions")
    parser.add_argument('--model2_label', default='Model 2',
                       help="Legend label for the second model's predictions (if provided)")
    parser.add_argument('--split_12h', action='store_true',
                       help='Split plots into 12-hour windows and generate multiple plots')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.df_results_path):
        print(f"Error: Results file not found: {args.df_results_path}")
        sys.exit(1)
    
    if not os.path.exists(args.df_metrics_path):
        print(f"Error: Metrics file not found: {args.df_metrics_path}")
        sys.exit(1)
    if args.df_results_path_2 is not None and not os.path.exists(args.df_results_path_2):
        print(f"Error: Second results file/folder not found: {args.df_results_path_2}")
        sys.exit(1)
    
    try:
        # Load and prepare data
        df_results, df_metrics = load_and_prepare_data(args.df_results_path, args.df_metrics_path)
        df_results_model2 = None
        if args.df_results_path_2:
            df_model2_raw, model2_kind = _load_model2_input(args.df_results_path_2)
            if model2_kind == 'results':
                df_results_model2 = df_model2_raw
            else:
                # metrics: compute mean/max per 5-minute window
                # Ensure previous metrics were added first (df_results_final uses datetime)
                pass
        
        # Add previous metrics
        df_results_final = add_previous_metrics(df_results, df_metrics)
        if args.df_results_path_2:
            if model2_kind == 'results':
                df_results_final = merge_second_model_predictions(df_results_final, df_results_model2)
            else:
                df_results_final = add_model2_from_metrics(df_results_final, df_model2_raw)
        
        # Clean dataframe
        df_results_final = clean_dataframe(df_results_final)
        
        # Print data summary to verify functionality
        print_data_summary(df_results_final)
        
        # Create comparison plot(s)
        if not args.split_12h:
            _create_single_plot(df_results_final, args.output_image_path, args.model1_label, args.model2_label if args.df_results_path_2 else None)
        else:
            _create_12h_plots(df_results_final, args.output_image_path, args.model1_label, args.model2_label if args.df_results_path_2 else None)
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()