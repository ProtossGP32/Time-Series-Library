#!/usr/bin/env python3
"""
Model Improvement Analysis Script

This script replicates the functionality of model_improvement.ipynb.
It compares model predictions vs actual values by processing MLflow results
and raw metrics data to generate comparison plots.

Usage:
    python model_improvement_analysis.py <df_results_path> <df_metrics_path> <output_image_path>

Arguments:
    df_results_path: Path to the MLflow results CSV file
    df_metrics_path: Path to the raw metrics CSV file  
    output_image_path: Path where to save the comparison plot
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import os
import sys


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
    
    # Load results - handle both file and folder
    if os.path.isdir(df_results_path):
        print(f"Loading multiple CSV files from folder: {df_results_path}")
        csv_files = [f for f in os.listdir(df_results_path) 
                    if f.endswith('.csv') and '_mlflow_results' in f]
        
        if not csv_files:
            raise ValueError(f"No MLflow results CSV files found in directory: {df_results_path}")
        
        print(f"Found {len(csv_files)} MLflow results CSV files")
        
        # Read and combine all CSV files
        dfs = []
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


def _create_single_plot(df_data, save_path):
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
             label='Predicted Mean', linewidth=2, color='red', marker='s', markersize=3)
    
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
             label='Predicted Max', linewidth=2, color='red', marker='s', markersize=3)
    
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


def _create_12h_plots(df_data, save_path):
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
            _create_window_plot(window_data, window_filename, current_time, window_end, plot_count)
        
        current_time = window_end
    
    print(f"Created {plot_count} 12-hour window plots")


def _create_window_plot(df_data, save_path, window_start, window_end, window_num):
    """Create a plot for a specific 12-hour window."""
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Mean values
    ax1.plot(df_data['start_time'],
             df_data['previous_mean'],
             label='Previous Mean', linewidth=2, color='blue', marker='o', markersize=4)
    ax1.plot(df_data['start_time'],
             df_data['avg_qos_0'],
             label='Predicted Mean', linewidth=2, color='red', marker='s', markersize=4)
    
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
             label='Predicted Max', linewidth=2, color='red', marker='s', markersize=4)
    
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


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(
        description='Model Improvement Analysis - Compare predictions vs actual values',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python model_improvement_analysis.py results.csv metrics.csv output.png
    python model_improvement_analysis.py /path/to/results_folder/ /path/to/metrics.csv /path/to/output.png
    python model_improvement_analysis.py /path/to/results.csv /path/to/metrics.csv /path/to/output.png
        """
    )
    
    parser.add_argument('df_results_path', 
                       help='Path to the MLflow results CSV file or folder containing multiple CSV files')
    parser.add_argument('df_metrics_path', 
                       help='Path to the raw metrics CSV file or folder containing multiple CSV files')
    parser.add_argument('output_image_path', 
                       help='Path where to save the comparison plot')
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
    
    try:
        # Load and prepare data
        df_results, df_metrics = load_and_prepare_data(args.df_results_path, args.df_metrics_path)
        
        # Add previous metrics
        df_results_final = add_previous_metrics(df_results, df_metrics)
        
        # Clean dataframe
        df_results_final = clean_dataframe(df_results_final)
        
        # Print data summary to verify functionality
        print_data_summary(df_results_final)
        
        # Create comparison plot(s)
        plot_predictions_vs_previous(df_results_final, args.output_image_path, args.split_12h)
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()