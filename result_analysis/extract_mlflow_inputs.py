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


def report_timestamp_overlaps(df_metrics, output_dir, examples_limit=10):
    """Analyze and report timestamps with repeated samples across metrics files.

    Prints a short summary and example rows. Does not write any files.
    """

    counts = df_metrics.groupby('date').size().rename('rows').reset_index()

    merged = counts
    has_cluster = 'cluster' in df_metrics.columns
    if has_cluster:
        unique_clusters = df_metrics.groupby('date')['cluster'].nunique().rename('unique_clusters').reset_index()
        merged = merged.merge(unique_clusters, on='date', how='left')

        # Per-timestamp cluster counts as JSON for compactness
        per_cluster = (
            df_metrics.groupby(['date', 'cluster']).size().rename('count').reset_index()
        )
        per_cluster_map = (
            per_cluster.groupby('date')
            .apply(lambda g: {str(c): int(n) for c, n in zip(g['cluster'], g['count'])})
            .rename('cluster_counts')
            .reset_index()
        )
        merged = merged.merge(per_cluster_map, on='date', how='left')

    # Print summary
    total_ts = len(merged)
    repeated_ts = int((merged['rows'] > 1).sum())
    print(f"Total timestamps: {total_ts}")
    print(f"Timestamps with >1 row: {repeated_ts}")
    if has_cluster and 'unique_clusters' in merged.columns:
        repeated_multi_cluster = int((merged['unique_clusters'] > 1).sum())
        print(f"Timestamps with >1 unique cluster: {repeated_multi_cluster}")

    # Show a few examples
    examples = merged[merged['rows'] > 1].sort_values('date').head(examples_limit)
    if not examples.empty:
        print("Examples of overlapping timestamps:")
        for _, r in examples.iterrows():
            line = f"  {r['date']} | rows={int(r['rows'])}"
            if has_cluster and 'unique_clusters' in merged.columns:
                line += f", unique_clusters={int(r['unique_clusters'])}"
                if 'cluster_counts' in r and isinstance(r['cluster_counts'], dict):
                    line += f", cluster_counts={json.dumps(r['cluster_counts'])}"
            print(line)

        # Also print feature rows for those example timestamps (both clusters)
        print("\nFeature rows for example overlapping timestamps:")
        feature_cols_priority = [
            'pipelines_status_avg_fps',
            'pipelines_status_avg_pipeline_latency',
            'node_cpu_usage',
            'node_mem_usage',
            'pipelines_server_cpu_usage',
            'pipelines_server_mem_usage',
            'number_pipelines',
            'pipelines_status_realtime_pipeline_latency',
        ]
        example_dates = list(examples['date'])
        for ts in example_dates:
            ts_rows = df_metrics[df_metrics['date'] == ts]
            # Determine columns to print: prefer the priority list if present, else all except raw 'date'
            available_priority = [c for c in feature_cols_priority if c in ts_rows.columns]
            print_cols = ['date']
            if 'cluster' in ts_rows.columns:
                print_cols.append('cluster')
            print_cols.extend(available_priority if available_priority else [c for c in ts_rows.columns if c != 'date'])
            # Print compact lines
            for _, rr in ts_rows.iterrows():
                data_map = {c: rr[c] for c in print_cols if c in rr}
                print(f"  - {json.dumps(data_map, default=str)}")
    


def prefer_cluster_on_duplicate_timestamps(df_metrics, preferred_cluster_id):
    """When multiple rows share the same timestamp, keep only the preferred cluster rows.

    - Prints the number of timestamps with repeated samples (any cluster).
    - If a repeated timestamp contains the preferred cluster, drops other clusters for that timestamp.
    - If a repeated timestamp does not contain the preferred cluster, leaves it as-is.
    """
    if 'date' not in df_metrics.columns:
        return df_metrics

    df = df_metrics.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Count timestamps with >1 rows
    repeated_mask = df.duplicated('date', keep=False)
    repeated_dates = df.loc[repeated_mask, 'date'].drop_duplicates()
    print(f"Repeated timestamps (any cluster): {len(repeated_dates)}")

    if 'cluster' not in df.columns or len(repeated_dates) == 0:
        return df

    # Dates where preferred cluster is present
    has_preferred = df[(df['cluster'] == preferred_cluster_id) & (df['date'].isin(repeated_dates))]['date'].drop_duplicates()

    # Keep rows that are not repeated OR are preferred cluster at repeated dates
    keep_mask = (~repeated_mask) | ((df['cluster'] == preferred_cluster_id) & (df['date'].isin(has_preferred)))
    df_filtered = df[keep_mask]
    return df_filtered


def extract_5min_windows(df_results, df_metrics, output_dir):
    """For each result timestamp, extract the previous 10 samples before it.

    This does NOT enforce a fixed duration window. It simply takes the last
    10 chronologically prior samples (date < prediction_time)."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert timestamps
    df_results['start_time_dt'] = pd.to_datetime(df_results['start_time'], unit='s')
    df_metrics['date'] = pd.to_datetime(df_metrics['date'])
    
    # Sort results by timestamp
    df_results = df_results.sort_values('start_time_dt').reset_index(drop=True)
    
    metadata = []
    created_count = 0
    skipped_count = 0
    skipped_less_than_10 = 0
    skipped_more_than_10 = 0
    
    for idx, row in df_results.iterrows():
        # Get prediction timestamp rounded down to the nearest 30-second boundary
        start_dt = row['start_time_dt']
        if start_dt.second >= 30:
            prediction_time = start_dt.replace(second=30, microsecond=0)
        else:
            prediction_time = start_dt.replace(second=0, microsecond=0)
        
        # Collect the previous 10 samples strictly before the prediction time
        history_data = df_metrics[df_metrics['date'] < prediction_time]
        history_data = history_data.sort_values('date')
        window_data = history_data.tail(10)

        # Note: cluster filtering intentionally disabled to observe repetitions across files
        
        # Check if we have exactly 10 samples
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
                'window_start': window_data['date'].iloc[0].isoformat(),
                'window_end': window_data['date'].iloc[-1].isoformat(),
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
            if len(window_data) < 10:
                skipped_less_than_10 += 1
            elif len(window_data) > 10:
                skipped_more_than_10 += 1
    
    print(f"Created: {created_count} windows (10 time steps each)")
    print(f"Skipped: {skipped_count} windows (wrong number of time steps)")
    if skipped_count:
        print(f"  - <10 samples: {skipped_less_than_10}")
        print(f"  - >10 samples: {skipped_more_than_10}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Extract 5-minute input windows')
    parser.add_argument('df_results_path', help='Path to MLflow results CSV')
    parser.add_argument('df_metrics_path', help='Path to raw metrics CSV')
    parser.add_argument('output_dir', help='Directory to save extracted windows')
    # Note: cluster filtering removed by request
    parser.add_argument('--report_overlaps', action='store_true',
                        help='Analyze repeated timestamps across metrics and print diagnostics (no files written)')
    parser.add_argument('--overlap_examples', type=int, default=10,
                        help='Number of example repeated timestamps to print')
    
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

        # Report overlaps on the raw metrics if requested
        if args.report_overlaps:
            report_timestamp_overlaps(df_metrics, args.output_dir, args.overlap_examples)

        # Deduplicate by preferring a specific cluster on repeated timestamps
        preferred_cluster_id = 'fd7816db-7948-4602-af7a-1d51900792a7'
        df_metrics = prefer_cluster_on_duplicate_timestamps(df_metrics, preferred_cluster_id)
        metadata = extract_5min_windows(
            df_results,
            df_metrics,
            args.output_dir,
        )
        print(f"Output saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()