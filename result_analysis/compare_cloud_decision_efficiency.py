#!/usr/bin/env python3
"""
Compare two approaches to quantify seconds where Approach 2 avoids SLA
violations by running in Cloud while Approach 1 is in Edge and above the SLA.

Input layout (same as other analysis scripts):
  <root>/<approach>/<job>/preprocessed/*.csv  (metrics with 'date', 'cluster')
  <root>/<approach>/<job>/raw/*.json          (each JSON has start_time/end_time)

Outputs:
  - detailed_pairs.csv: One row per experiment pair (aligned by step index)
  - summary.csv: Aggregated totals and fractions across all pairs

Usage:
  python compare_cloud_decision_efficiency.py \
    --root /home/jolivera/Documents/CloudSkin/Time-Series-Library/dataset/comparison_of_approaches \
    --job job-0 \
    [--approach_1 reactive] \
    [--approach_2 random_forest] \
    [--sla 0.2] \
    [--out_dir result_analysis/cloud_decision_efficiency]
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Column expected for latency
TARGET_COLUMN = 'pipelines_status_realtime_pipeline_latency'

# Cloud/Edge cluster ids (consistent across the dataset and other scripts)
CLOUD_CLUSTER_ID = 'fd7816db-7948-4602-af7a-1d51900792a7'
EDGE_CLUSTER_ID = 'eb0e3eaa-b668-4ad6-bc10-2bb0eb7da259'


@dataclass
class ExperimentWindow:
    start_epoch_seconds: float
    end_epoch_seconds: float
    source_file: str


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
    for required in ('date', TARGET_COLUMN, 'cluster'):
        if required not in df_all.columns:
            raise ValueError(f"Expected '{required}' column in preprocessed CSVs under {preprocessed_dir}")
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

    experiments.sort(key=lambda e: e.start_epoch_seconds)
    return experiments


def _slice_experiment(df_metrics: pd.DataFrame, window: ExperimentWindow) -> pd.DataFrame:
    start_time = pd.to_datetime(window.start_epoch_seconds, unit='s')
    end_time = pd.to_datetime(window.end_epoch_seconds, unit='s')
    df_exp = df_metrics[(df_metrics['date'] >= start_time) & (df_metrics['date'] <= end_time)].copy()
    df_exp = df_exp.sort_values('date').reset_index(drop=True)
    df_exp['step_index'] = np.arange(len(df_exp), dtype=int)
    df_exp['time_since_start_seconds'] = (df_exp['date'] - start_time).dt.total_seconds()
    return df_exp


def _align_by_step_and_measure(
    df_1: pd.DataFrame,
    df_2: pd.DataFrame,
    sla: float,
    debug: bool = False,
) -> Dict[str, float]:
    # Select relevant columns and rename
    a1 = df_1[['time_since_start_seconds', TARGET_COLUMN, 'cluster']].copy()
    a1.rename(columns={TARGET_COLUMN: 'latency_1', 'cluster': 'cluster_1'}, inplace=True)
    a2 = df_2[[TARGET_COLUMN, 'cluster']].copy()
    a2.rename(columns={TARGET_COLUMN: 'latency_2', 'cluster': 'cluster_2'}, inplace=True)

    if a1.empty or a2.empty:
        return {
            'overlap_seconds_total': 0.0,
            'approach_1_above_edge_seconds': 0.0,
            'approach_2_cloud_seconds': 0.0,
            'avoided_seconds': 0.0,
            'not_avoided_seconds_2_also_above': 0.0,
            'num_overlap_rows': 0,
        }

    # Step-index alignment: trim both to the shorter length
    min_len = int(min(len(a1), len(a2)))
    if debug:
        print(f"Step alignment lengths: approach_1={len(a1)}, approach_2={len(a2)}, using min_len={min_len}")
    if min_len == 0:
        return {
            'overlap_seconds_total': 0.0,
            'approach_1_above_edge_seconds': 0.0,
            'approach_2_cloud_seconds': 0.0,
            'avoided_seconds': 0.0,
            'not_avoided_seconds_2_also_above': 0.0,
            'num_overlap_rows': 0,
        }

    a1 = a1.iloc[:min_len].reset_index(drop=True)
    a2 = a2.iloc[:min_len].reset_index(drop=True)

    # Compute dt_next on approach_1 timeline
    t = a1['time_since_start_seconds'].astype(float)
    dt_next = t.shift(-1) - t
    dt_next = dt_next.fillna(0.0)

    a1_above = a1['latency_1'].astype(float) >= sla
    a1_edge = a1['cluster_1'].astype(str) == EDGE_CLUSTER_ID
    a2_cloud = a2['cluster_2'].astype(str) == CLOUD_CLUSTER_ID
    a2_below = a2['latency_2'].astype(float) < sla
    a2_above = ~a2_below

    dt = dt_next.astype(float)
    overlap_seconds_total = float(dt.sum())
    approach_1_above_edge_seconds = float(((a1_above & a1_edge).astype(float) * dt).sum())
    approach_2_cloud_seconds = float((a2_cloud.astype(float) * dt).sum())
    avoided_seconds = float(((a1_above & a1_edge & a2_cloud & a2_below).astype(float) * dt).sum())
    not_avoided_seconds_2_also_above = float(((a1_above & a1_edge & a2_cloud & a2_above).astype(float) * dt).sum())

    return {
        'overlap_seconds_total': overlap_seconds_total,
        'approach_1_above_edge_seconds': approach_1_above_edge_seconds,
        'approach_2_cloud_seconds': approach_2_cloud_seconds,
        'avoided_seconds': avoided_seconds,
        'not_avoided_seconds_2_also_above': not_avoided_seconds_2_also_above,
        'num_overlap_rows': min_len,
    }


def _prepare_experiments_for_approach(approach_path: str, job: str) -> Tuple[List[ExperimentWindow], pd.DataFrame]:
    preprocessed_dir = os.path.join(approach_path, job, 'preprocessed')
    raw_dir = os.path.join(approach_path, job, 'raw')
    df_metrics = _load_preprocessed_metrics(preprocessed_dir)
    windows = _load_experiment_windows(raw_dir)
    return windows, df_metrics


def main():
    parser = argparse.ArgumentParser(description='Measure cloud decision efficiency between two approaches')
    parser.add_argument('--root', required=True, help='Path to comparison_of_approaches root directory')
    parser.add_argument('--job', default='job-0', help='Job subdirectory to process (default: job-0)')
    parser.add_argument('--approach_1', default='reactive', help='First approach subfolder (reference for time attribution)')
    parser.add_argument('--approach_2', default='random_forest', help='Second approach subfolder (compared approach)')
    parser.add_argument('--sla', type=float, default=0.2, help='SLA threshold for latency (default: 0.2)')
    parser.add_argument('--out_dir', default='result_analysis/cloud_decision_efficiency', help='Directory to write outputs')
    parser.add_argument('--debug', action='store_true', help='Print debug information about windows and alignment')
    args = parser.parse_args()

    root = args.root
    if not os.path.isdir(root):
        raise NotADirectoryError(f"Root not found: {root}")

    a1_path = os.path.join(root, args.approach_1)
    a2_path = os.path.join(root, args.approach_2)
    if not os.path.isdir(a1_path):
        raise NotADirectoryError(f"Approach 1 folder not found: {a1_path}")
    if not os.path.isdir(a2_path):
        raise NotADirectoryError(f"Approach 2 folder not found: {a2_path}")

    # Load experiments and metrics
    a1_windows, a1_metrics = _prepare_experiments_for_approach(a1_path, args.job)
    a2_windows, a2_metrics = _prepare_experiments_for_approach(a2_path, args.job)

    if args.debug:
        print(f"{args.approach_1} windows ({len(a1_windows)}):")
        for w in a1_windows:
            print(f"  {w.source_file}: {w.start_epoch_seconds} -> {w.end_epoch_seconds}")
        print(f"{args.approach_2} windows ({len(a2_windows)}):")
        for w in a2_windows:
            print(f"  {w.source_file}: {w.start_epoch_seconds} -> {w.end_epoch_seconds}")

    pair_rows: List[Dict[str, object]] = []

    for w1 in a1_windows:
        df1 = _slice_experiment(a1_metrics, w1)
        if df1.empty:
            continue

        for w2 in a2_windows:
            df2 = _slice_experiment(a2_metrics, w2)
            if df2.empty:
                continue

            measures = _align_by_step_and_measure(df1, df2, sla=args.sla, debug=args.debug)
            if measures['num_overlap_rows'] == 0:
                if args.debug:
                    print(f"No aligned steps for pair {w1.source_file} vs {w2.source_file}")
                continue

            row: Dict[str, object] = {
                'approach_1': args.approach_1,
                'approach_2': args.approach_2,
                'approach_1_experiment': w1.source_file,
                'approach_2_experiment': w2.source_file,
                'overlap_seconds_total': measures['overlap_seconds_total'],
                'approach_1_above_edge_seconds': measures['approach_1_above_edge_seconds'],
                'approach_2_cloud_seconds': measures['approach_2_cloud_seconds'],
                'avoided_seconds': measures['avoided_seconds'],
                'not_avoided_seconds_2_also_above': measures['not_avoided_seconds_2_also_above'],
                'avoided_fraction_of_approach_1_above_edge': (measures['avoided_seconds'] / measures['approach_1_above_edge_seconds']) if measures['approach_1_above_edge_seconds'] > 0.0 else np.nan,
                'num_overlap_rows': measures['num_overlap_rows'],
            }
            pair_rows.append(row)

    if not pair_rows:
        raise RuntimeError('No experiment pairs found with alignable steps.')

    df_pairs = pd.DataFrame(pair_rows)

    # Aggregated summary
    totals = {
        'approach_1': args.approach_1,
        'approach_2': args.approach_2,
        'pairs_count': int(df_pairs.shape[0]),
        'overlap_seconds_total': float(df_pairs['overlap_seconds_total'].sum()),
        'approach_1_above_edge_seconds': float(df_pairs['approach_1_above_edge_seconds'].sum()),
        'approach_2_cloud_seconds': float(df_pairs['approach_2_cloud_seconds'].sum()),
        'avoided_seconds': float(df_pairs['avoided_seconds'].sum()),
        'not_avoided_seconds_2_also_above': float(df_pairs['not_avoided_seconds_2_also_above'].sum()),
    }
    totals['avoided_fraction_of_approach_1_above_edge'] = (
        totals['avoided_seconds'] / totals['approach_1_above_edge_seconds']
        if totals['approach_1_above_edge_seconds'] > 0.0 else np.nan
    )

    df_summary = pd.DataFrame([totals])

    # Write outputs
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    pairs_out = os.path.join(out_dir, 'detailed_pairs.csv')
    summary_out = os.path.join(out_dir, 'summary.csv')
    df_pairs.to_csv(pairs_out, index=False)
    df_summary.to_csv(summary_out, index=False)
    print(f"Wrote detailed pairs to {pairs_out}")
    print(f"Wrote summary to {summary_out}")


if __name__ == '__main__':
    main()


