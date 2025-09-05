#!/usr/bin/env python3
"""
Aggregate repeated experiments per approach by aligning on time delta since
experiment start, and compute mean and max of real-time pipeline latency over
time across experiments.

Usage:
    python aggregate_experiments_by_time_delta.py \
        --root /home/jolivera/Documents/CloudSkin/Time-Series-Library/dataset/comparison_of_approaches \
        --job job-0 \
        [--approach informer] \
        [--output_subdir aggregated]

Output:
    For each processed approach, writes a CSV:
        <root>/<approach>/<job>/<output_subdir>/aggregated_realtime_latency_by_time_delta.csv

Notes:
    - Preprocessed CSVs are expected under <root>/<approach>/<job>/preprocessed/*.csv
    - Raw JSON experiment descriptors are expected under <root>/<approach>/<job>/raw/*.json
    - Alignment is by ordinal step within each experiment (step_index), avoiding
      absolute timestamps. time_since_start_seconds is derived from median step
      seconds across that approach's experiments.
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ExperimentWindow:
    start_epoch_seconds: float
    end_epoch_seconds: float
    source_file: str


def _discover_approaches(root: str) -> List[str]:
    return [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]


def _load_preprocessed_metrics(preprocessed_dir: str) -> pd.DataFrame:
    csv_files = [
        os.path.join(preprocessed_dir, f)
        for f in os.listdir(preprocessed_dir)
        if f.endswith('.csv')
    ]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in preprocessed dir: {preprocessed_dir}")

    dataframes: List[pd.DataFrame] = []
    for path in sorted(csv_files):
        df = pd.read_csv(path)
        dataframes.append(df)

    df_all = pd.concat(dataframes, ignore_index=True)
    if 'date' not in df_all.columns:
        raise ValueError("Expected 'date' column in preprocessed CSVs")
    df_all['date'] = pd.to_datetime(df_all['date'])
    return df_all.sort_values('date').reset_index(drop=True)


def _load_experiment_windows(raw_dir: str) -> List[ExperimentWindow]:
    json_files = [
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if f.endswith('.json')
    ]
    if not json_files:
        raise FileNotFoundError(f"No JSON experiment descriptors found in raw dir: {raw_dir}")

    experiments: List[ExperimentWindow] = []
    for path in sorted(json_files):
        with open(path, 'r') as f:
            meta = json.load(f)
        # Allow both str and numeric epoch representations
        start_str = meta.get('start_time')
        end_str = meta.get('end_time')
        if start_str is None or end_str is None:
            raise ValueError(f"Missing start_time/end_time in {path}")
        try:
            start_epoch = float(start_str)
            end_epoch = float(end_str)
        except Exception as exc:
            raise ValueError(f"Invalid epoch in {path}: {exc}")
        experiments.append(ExperimentWindow(start_epoch_seconds=start_epoch, end_epoch_seconds=end_epoch, source_file=os.path.basename(path)))

    # Sort by start time to ensure chronological ordering
    experiments.sort(key=lambda e: e.start_epoch_seconds)
    return experiments


def _slice_experiment(df_metrics: pd.DataFrame, window: ExperimentWindow) -> pd.DataFrame:
    start_time = pd.to_datetime(window.start_epoch_seconds, unit='s')
    end_time = pd.to_datetime(window.end_epoch_seconds, unit='s')
    df_exp = df_metrics[(df_metrics['date'] >= start_time) & (df_metrics['date'] <= end_time)].copy()
    # Order and add step_index (ordinal within experiment)
    df_exp = df_exp.sort_values('date').reset_index(drop=True)
    df_exp['step_index'] = np.arange(len(df_exp), dtype=int)
    # Compute per-row delta seconds from actual start
    df_exp['time_since_start_seconds'] = (df_exp['date'] - start_time).dt.total_seconds()
    return df_exp


def _infer_step_seconds(df_exp: pd.DataFrame) -> Optional[float]:
    if df_exp.shape[0] < 2:
        return None
    diffs = df_exp['date'].diff().dt.total_seconds().dropna()
    if diffs.empty:
        return None
    # Use median to be robust to outliers
    return float(np.median(diffs.values))


def aggregate_approach(
    approach_path: str,
    job: str,
    output_subdir: str,
) -> str:
    preprocessed_dir = os.path.join(approach_path, job, 'preprocessed')
    raw_dir = os.path.join(approach_path, job, 'raw')
    output_dir = os.path.join(approach_path, job, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    df_metrics = _load_preprocessed_metrics(preprocessed_dir)
    windows = _load_experiment_windows(raw_dir)

    per_experiment: List[pd.DataFrame] = []
    step_seconds_samples: List[float] = []
    for window in windows:
        df_exp = _slice_experiment(df_metrics, window)
        if df_exp.empty:
            # Skip experiments with no metric rows
            continue
        step_seconds = _infer_step_seconds(df_exp)
        if step_seconds is not None:
            step_seconds_samples.append(step_seconds)
        df_exp['experiment_id'] = window.source_file
        per_experiment.append(df_exp[['experiment_id', 'step_index', 'time_since_start_seconds', 'pipelines_status_realtime_pipeline_latency']])

    if not per_experiment:
        raise RuntimeError(f"No experiments with data found for {approach_path}")

    df_all_steps = pd.concat(per_experiment, ignore_index=True)

    # Use median step seconds across experiments; default to 30 if unknown
    median_step_seconds = float(np.median(step_seconds_samples)) if step_seconds_samples else 30.0

    # Aggregate across experiments at the same step_index
    grouped = df_all_steps.groupby('step_index')
    agg_df = grouped['pipelines_status_realtime_pipeline_latency'].agg(
        mean_over_experiments='mean',
        max_over_experiments='max',
        count='count',
    ).reset_index()
    agg_df['time_since_start_seconds'] = agg_df['step_index'] * median_step_seconds

    # Order columns for readability
    agg_df = agg_df[['step_index', 'time_since_start_seconds', 'mean_over_experiments', 'max_over_experiments', 'count']]

    output_csv = os.path.join(output_dir, 'aggregated_realtime_latency_by_time_delta.csv')
    agg_df.to_csv(output_csv, index=False)
    return output_csv


def main():
    parser = argparse.ArgumentParser(description='Aggregate experiments by time delta for each approach')
    parser.add_argument('--root', required=True, help='Path to comparison_of_approaches root directory')
    parser.add_argument('--job', default='job-0', help='Job subdirectory to process (default: job-0)')
    parser.add_argument('--approach', default=None, help='If provided, only process this approach (subfolder name)')
    parser.add_argument('--output_subdir', default='aggregated', help='Subdirectory under job where to write outputs')
    args = parser.parse_args()

    root = args.root
    if not os.path.isdir(root):
        raise NotADirectoryError(f"Root not found: {root}")

    approaches = [args.approach] if args.approach else _discover_approaches(root)
    results: Dict[str, str] = {}

    for approach in sorted(approaches):
        approach_path = os.path.join(root, approach)
        job_path = os.path.join(approach_path, args.job)
        if not os.path.isdir(job_path):
            # Skip folders without the job subdir
            continue
        try:
            output_csv = aggregate_approach(approach_path, args.job, args.output_subdir)
            results[approach] = output_csv
            print(f"Aggregated {approach} -> {output_csv}")
        except Exception as exc:
            print(f"Failed to aggregate {approach}: {exc}")

    if not results:
        raise RuntimeError('No approaches were processed. Check paths and inputs.')


if __name__ == '__main__':
    main()


