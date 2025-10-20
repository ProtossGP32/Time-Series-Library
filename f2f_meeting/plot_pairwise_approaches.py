#!/usr/bin/env python3
"""
Plot a pairwise comparison between one experiment from Approach 1 and one from
Approach 2, aligned by step index (not timestamps). Shade segments where
Approach 2 avoids SLA violations while Approach 1 is above SLA on Edge.

Input layout (same as other analysis scripts):
  <root>/<approach>/<job>/preprocessed/*.csv  (metrics with 'date', 'cluster')
  <root>/<approach>/<job>/raw/*.json          (each JSON has start_time/end_time)

Usage example:
  python plot_pairwise_approaches.py \
    --root /home/jolivera/Documents/CloudSkin/Time-Series-Library/dataset/comparison_of_approaches \
    --job job-0 \
    --approach_1 reactive \
    --approach_2 random_forest \
    --experiment_1 <approach_1_json> \
    --experiment_2 <approach_2_json> \
    --sla 0.2 \
    --out_dir result_analysis/pair_plots
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


TARGET_COLUMN = 'pipelines_status_realtime_pipeline_latency'
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
            raise ValueError(f"Expected '{required}' in preprocessed CSVs under {preprocessed_dir}")
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


def _find_window_by_name(windows: List[ExperimentWindow], name: str) -> ExperimentWindow:
    for w in windows:
        if w.source_file == name:
            return w
    raise FileNotFoundError(f"Experiment JSON '{name}' not found among: {[w.source_file for w in windows]}")


def _contiguous_runs(mask: pd.Series) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    if mask.empty:
        return runs
    in_run = False
    start = 0
    for i, val in enumerate(mask.astype(bool).values.tolist()):
        if val and not in_run:
            in_run = True
            start = i
        elif not val and in_run:
            in_run = False
            runs.append((start, i - 1))
    if in_run:
        runs.append((start, len(mask) - 1))
    return runs


def main():
    parser = argparse.ArgumentParser(description='Pairwise plot by step index: shade indices where Approach 2 avoids SLA vs Approach 1')
    parser.add_argument('--root', required=True, help='Path to comparison_of_approaches root directory')
    parser.add_argument('--job', default='job-0', help='Job subdirectory to process (default: job-0)')
    parser.add_argument('--approach_1', default='reactive', help='First approach subfolder')
    parser.add_argument('--approach_2', default='random_forest', help='Second approach subfolder')
    parser.add_argument('--experiment_1', required=True, help='Exact JSON filename in approach_1 raw dir')
    parser.add_argument('--experiment_2', required=True, help='Exact JSON filename in approach_2 raw dir')
    parser.add_argument('--sla', type=float, default=0.2, help='SLA threshold for latency; default 0.2')
    parser.add_argument('--out_dir', default='result_analysis/pair_plots', help='Directory to write the plot image')
    args = parser.parse_args()

    a1_path = os.path.join(args.root, args.approach_1)
    a2_path = os.path.join(args.root, args.approach_2)

    # Load metrics and experiment windows
    a1_metrics = _load_preprocessed_metrics(os.path.join(a1_path, args.job, 'preprocessed'))
    a1_windows = _load_experiment_windows(os.path.join(a1_path, args.job, 'raw'))
    a2_metrics = _load_preprocessed_metrics(os.path.join(a2_path, args.job, 'preprocessed'))
    a2_windows = _load_experiment_windows(os.path.join(a2_path, args.job, 'raw'))

    w1 = _find_window_by_name(a1_windows, args.experiment_1)
    w2 = _find_window_by_name(a2_windows, args.experiment_2)

    df1 = _slice_experiment(a1_metrics, w1)
    df2 = _slice_experiment(a2_metrics, w2)
    if df1.empty or df2.empty:
        raise RuntimeError('Selected experiments have no data after slicing their windows')

    # Align by step index
    min_len = int(min(len(df1), len(df2)))
    df1 = df1.iloc[:min_len].reset_index(drop=True)
    df2 = df2.iloc[:min_len].reset_index(drop=True)

    # Build plot data
    x = np.arange(min_len, dtype=int)
    y1 = df1[TARGET_COLUMN].astype(float).values
    y2 = df2[TARGET_COLUMN].astype(float).values
    c1 = df1['cluster'].astype(str)
    c2 = df2['cluster'].astype(str)

    # Avoided condition mask (Approach 2 avoids while Approach 1 violates on Edge)
    a1_above = y1 >= args.sla
    a1_edge = (c1 == EDGE_CLUSTER_ID).values
    a2_cloud = (c2 == CLOUD_CLUSTER_ID).values
    a2_below = y2 < args.sla
    avoided_mask = a1_above & a1_edge & a2_cloud & a2_below

    # Two subplots (shared x/y scales), shade avoided ranges
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, sharey=True)

    def _plot_by_cluster(ax, x_arr, y_arr, clusters):
        cloud_color = 'C0'
        edge_color = 'C1'
        run_start = 0
        for i in range(1, len(x_arr) + 1):
            end_run = (i == len(x_arr)) or (clusters.iloc[i] != clusters.iloc[i - 1])
            if end_run:
                seg_cluster = clusters.iloc[run_start]
                color = cloud_color if seg_cluster == CLOUD_CLUSTER_ID else edge_color
                ax.plot(x_arr[run_start:i], y_arr[run_start:i], color=color, linestyle='-', linewidth=2)
                run_start = i

    # Approach 1 subplot
    _plot_by_cluster(ax1, x, y1, c1)
    ax1.axhline(y=args.sla, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    for start, end in _contiguous_runs(pd.Series(avoided_mask)):
        ax1.axvspan(start, end, color='gold', alpha=0.25)
    ax1.set_title(f"{args.approach_1} Approach")
    ax1.set_ylabel('Realtime latency')
    ax1.grid(True, alpha=0.3)
    legend_elements = [
        Line2D([0], [0], color='C0', lw=2, label='Cloud'),
        Line2D([0], [0], color='C1', lw=2, label='Edge'),
        Line2D([0], [0], color='red', lw=1.5, linestyle='--', label='SLA'),
        Line2D([0], [0], color='gold', lw=6, alpha=0.25, label='Avoided SLA breach'),
    ]
    ax1.legend(handles=legend_elements, loc='best')

    # Approach 2 subplot
    _plot_by_cluster(ax2, x, y2, c2)
    ax2.axhline(y=args.sla, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    for start, end in _contiguous_runs(pd.Series(avoided_mask)):
        ax2.axvspan(start, end, color='gold', alpha=0.25)
    ax2.set_title(f"{args.approach_2} Approach")
    ax2.set_xlabel('Step index (aligned by min length)')
    ax2.set_ylabel('Realtime latency')
    ax2.grid(True, alpha=0.3)
    ax2.legend(handles=legend_elements, loc='best')

    fig.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    safe_1 = os.path.splitext(w1.source_file)[0]
    safe_2 = os.path.splitext(w2.source_file)[0]
    out_path = os.path.join(args.out_dir, f"pairplot_two_panels__{safe_1}__VS__{safe_2}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


if __name__ == '__main__':
    main()


