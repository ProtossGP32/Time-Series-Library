#!/usr/bin/env python3
"""
Iterate approaches and jobs, plot each experiment's latency evolution, and
change color segments at each rising-edge SLA crossing (False -> True for
latency >= SLA). Saves one image per experiment under a given subfolder inside
each approach/job.

Input layout is the same expected by the other analysis scripts:
  <root>/<approach>/<job>/preprocessed/*.csv   (metrics CSVs with 'date')
  <root>/<approach>/<job>/raw/*.json           (one per experiment with start_time/end_time)

Usage:
  python plot_experiments_with_sla_crossings.py \
    --root /home/jolivera/Documents/CloudSkin/Time-Series-Library/dataset/comparison_of_approaches \
    [--approach informer] \
    [--jobs job-0,job-1] \
    [--sla 0.2] \
    [--plot_subdir plots_sla_0_2]
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


TARGET_COLUMN = 'pipelines_status_realtime_pipeline_latency'


@dataclass
class ExperimentWindow:
    start_epoch_seconds: float
    end_epoch_seconds: float
    source_file: str


def _discover_approaches(root: str) -> List[str]:
    return [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]


def _discover_jobs(approach_path: str) -> List[str]:
    jobs: List[str] = []
    for name in os.listdir(approach_path):
        job_path = os.path.join(approach_path, name)
        if not os.path.isdir(job_path):
            continue
        # Heuristic: a valid job has both 'preprocessed' and 'raw'
        if os.path.isdir(os.path.join(job_path, 'preprocessed')) and os.path.isdir(os.path.join(job_path, 'raw')):
            jobs.append(name)
    return sorted(jobs)


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
    if TARGET_COLUMN not in df_all.columns:
        raise ValueError(f"Expected '{TARGET_COLUMN}' column in preprocessed CSVs")
    if 'cluster' not in df_all.columns:
        raise ValueError("Expected 'cluster' column in preprocessed CSVs")
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


def _segment_ids_by_rising_edges(above_series: pd.Series) -> pd.Series:
    """Return segment ids that increment at each rising edge (False->True)."""
    rising_edges = (above_series.astype(int).diff().fillna(0) == 1)
    # cumulative sum of rising edge booleans gives a new id per rising edge
    segment_ids = rising_edges.cumsum().astype(int)
    return segment_ids


def _plot_experiment(
    df_exp: pd.DataFrame,
    approach: str,
    job: str,
    experiment_id: str,
    sla: float,
    output_dir: str,
):
    t = df_exp['time_since_start_seconds'].astype(float) / 60.0  # minutes
    y = df_exp[TARGET_COLUMN].astype(float)
    above = y >= sla
    seg_ids = _segment_ids_by_rising_edges(above)
    num_crossings = int((above.astype(int).diff().fillna(0) == 1).sum())
    clusters = df_exp['cluster'].astype(str)

    plt.figure(figsize=(14, 6))

    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0', 'C1', 'C2', 'C3', 'C4'])

    # Determine cluster -> linestyle mapping (solid for first, dashed for second/others)
    unique_clusters = list(pd.unique(clusters))
    cluster_to_ls = {}
    if len(unique_clusters) == 0:
        cluster_to_ls = {}
    elif len(unique_clusters) == 1:
        cluster_to_ls[unique_clusters[0]] = '-'
    else:
        cluster_to_ls[unique_clusters[0]] = '-'
        cluster_to_ls[unique_clusters[1]] = '--'
        for c in unique_clusters[2:]:
            cluster_to_ls[c] = '--'

    # Plot runs where either SLA segment id or cluster changes
    run_start = 0
    for i in range(1, len(seg_ids) + 1):
        end_run = False
        if i == len(seg_ids):
            end_run = True
        else:
            if (seg_ids.iloc[i] != seg_ids.iloc[i - 1]) or (clusters.iloc[i] != clusters.iloc[i - 1]):
                end_run = True

        if end_run:
            color = colors[seg_ids.iloc[run_start] % len(colors)]
            cluster_value = clusters.iloc[run_start]
            linestyle = cluster_to_ls.get(cluster_value, '-')
            plt.plot(t.iloc[run_start:i], y.iloc[run_start:i], color=color, linestyle=linestyle, linewidth=2)
            run_start = i

    # SLA line
    plt.axhline(y=sla, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    plt.title(f"{approach} / {job} / {experiment_id}  |  crossings: {num_crossings}")
    plt.xlabel('Minutes since experiment start')
    plt.ylabel('Realtime latency')
    plt.grid(True, alpha=0.3)

    # Legend for line styles only (clusters)
    legend_elements = []
    added_styles = set()
    for c in unique_clusters:
        ls = cluster_to_ls.get(c, '-')
        if ls in added_styles:
            continue
        legend_elements.append(Line2D([0], [0], color='black', lw=2, linestyle=ls, label=c))
        added_styles.add(ls)
        if len(added_styles) >= 2:
            break
    if legend_elements:
        plt.legend(handles=legend_elements, loc='best')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    safe_id = os.path.splitext(experiment_id)[0]
    out_path = os.path.join(output_dir, f"{safe_id}_sla_{str(sla).replace('.', '_')}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot per-experiment latency with color changes at SLA rising-edge crossings')
    parser.add_argument('--root', required=True, help='Path to comparison_of_approaches root directory')
    parser.add_argument('--approach', default=None, help='If provided, only process this approach (subfolder name)')
    parser.add_argument('--jobs', default=None, help='Comma-separated list of jobs to process (default: discover all)')
    parser.add_argument('--sla', type=float, default=0.2, help='SLA threshold for latency; default 0.2')
    parser.add_argument('--plot_subdir', default='experiment_plots', help='Subfolder under each approach/job to write plots')
    args = parser.parse_args()

    root = args.root
    if not os.path.isdir(root):
        raise NotADirectoryError(f"Root not found: {root}")

    approaches = [args.approach] if args.approach else _discover_approaches(root)

    for approach in sorted(approaches):
        approach_path = os.path.join(root, approach)
        if not os.path.isdir(approach_path):
            continue

        if args.jobs:
            jobs = [j.strip() for j in args.jobs.split(',') if j.strip()]
        else:
            jobs = _discover_jobs(approach_path)

        for job in jobs:
            job_path = os.path.join(approach_path, job)
            preprocessed_dir = os.path.join(job_path, 'preprocessed')
            raw_dir = os.path.join(job_path, 'raw')
            output_dir = os.path.join(job_path, args.plot_subdir)

            if not (os.path.isdir(preprocessed_dir) and os.path.isdir(raw_dir)):
                continue

            try:
                df_metrics = _load_preprocessed_metrics(preprocessed_dir)
                windows = _load_experiment_windows(raw_dir)
            except Exception as exc:
                print(f"Skipping {approach}/{job}: {exc}")
                continue

            for window in windows:
                df_exp = _slice_experiment(df_metrics, window)
                if df_exp.empty:
                    continue
                _plot_experiment(df_exp, approach, job, window.source_file, args.sla, output_dir)


if __name__ == '__main__':
    main()


