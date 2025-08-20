#!/usr/bin/env python3
"""
Extract 5-minute input windows from MLflow experiment data.

Usage:
    python extract_input_windows.py <df_results_path> <df_metrics_path> <output_dir>
"""

import argparse
import pandas as pd
from datetime import timedelta
import os
import sys
import json


def load_mlflow_results(df_results_path):
    """Load MLflow results from file or folder."""
    if os.path.isdir(df_results_path):
        csv_files = [f for f in os.listdir(df_results_path) 
                    if f.endswith('.csv') and '_mlflow_results' in f]
        
        if not csv_files:
            raise ValueError(f"No MLflow results CSV files found in: {df_results_path}")
        
        dfs = []
        for filename in csv_files:
            filepath = os.path.join(df_results_path, filename)
            df = pd.read_csv(filepath, sep=';')
            dfs.append(df)
        
        df_results = pd.concat(dfs, ignore_index=True)
    else:
        df_results = pd.read_csv(df_results_path, sep=';')
    
    return df_results


def load_metrics_data(df_metrics_path):
    """Load raw metrics data from file or folder."""
    if os.path.isdir(df_metrics_path):
        csv_files = [f for f in os.listdir(df_metrics_path) if f.endswith('.csv')]
        
        if not csv_files:
            raise ValueError(f"No CSV files found in: {df_metrics_path}")
        
        dfs = []
        for filename in csv_files:
            filepath = os.path.join(df_metrics_path, filename)
            df = pd.read_csv(filepath)
            dfs.append(df)
        
        df_metrics = pd.concat(dfs, ignore_index=True).sort_values(by='date', na_position='last')
    else:
        df_metrics = pd.read_csv(df_metrics_path)
    
    return df_metrics


def extract_5min_windows(df_results, df_metrics, output_dir):
    """Extract 5-minute windows with exactly 10 time steps."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert timestamps
    df_results['start_time_dt'] = pd.to_datetime(df_results['start_time'], unit='s')
    df_metrics['date'] = pd.to_datetime(df_metrics['date'])
    
    # Sort results by timestamp
    df_results = df_results.sort_values('start_time_dt').reset_index(drop=True)
    
    metadata = []
    created_count = 0
    skipped_count = 0
    
    for idx, row in df_results.iterrows():
        # Get prediction timestamp (rounded to minute)
        prediction_time = row['start_time_dt'].replace(second=0, microsecond=0)
        
        # Define 5-minute window
        window_end = prediction_time
        window_start = prediction_time - timedelta(minutes=5)
        
        # Extract metrics data from this window - keep ALL original columns
        window_data = df_metrics[
            (df_metrics['date'] >= window_start) & 
            (df_metrics['date'] < window_end)
        ]
        
        # Check if we have exactly 10 time steps
        if len(window_data) == 10:
            # Sort by date but keep all original columns intact
            window_data = window_data.sort_values('date').reset_index(drop=True)
            
            # Create filename and save with exact same structure as df_metrics
            filename = f"window_{prediction_time.strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = os.path.join(output_dir, filename)
            window_data.to_csv(filepath, index=False)
            
            created_count += 1
            
            # Add to metadata
            meta_entry = {
                'window_id': idx,
                'prediction_timestamp': prediction_time.isoformat(),
                'window_start': window_start.isoformat(),
                'window_end': window_end.isoformat(),
                'filename': filename,
                'original_start_time': row['start_time'],
            }
            
            # Add original predictions for reference
            for col in ['avg_qos_0', 'max_qos_0', 'qos_0', 
                       'real_future_values_5_min_mean', 'real_future_values_5_min_max']:
                if col in row:
                    meta_entry[f'original_{col}'] = row[col]
            
            metadata.append(meta_entry)
        else:
            skipped_count += 1
    
    print(f"Created: {created_count} windows (10 time steps each)")
    print(f"Skipped: {skipped_count} windows (wrong number of time steps)")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Extract 5-minute input windows')
    parser.add_argument('df_results_path', help='Path to MLflow results CSV')
    parser.add_argument('df_metrics_path', help='Path to raw metrics CSV')
    parser.add_argument('output_dir', help='Directory to save extracted windows')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.df_results_path):
        print(f"Error: MLflow results not found: {args.df_results_path}")
        sys.exit(1)
    
    if not os.path.exists(args.df_metrics_path):
        print(f"Error: Metrics data not found: {args.df_metrics_path}")
        sys.exit(1)
    
    try:
        df_results = load_mlflow_results(args.df_results_path)
        df_metrics = load_metrics_data(args.df_metrics_path)
        metadata = extract_5min_windows(df_results, df_metrics, args.output_dir)
        print(f"Output saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()