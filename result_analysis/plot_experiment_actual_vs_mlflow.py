#!/usr/bin/env python3
"""
Plot actual metrics vs MLflow predictions for a single experiment.

We locate the experiment's window (start/end) from its raw JSON. The same
JSON is expected to contain MLflow results with prediction timestamps and
values under the field 'avg_qos_0'. We then:
  - Load actual metrics from preprocessed CSVs and slice to the window
  - Extract MLflow predictions (timestamps, values) from the experiment JSON
  - Plot both series together on the same time axis

Experiment JSON must include:
  - start_time, end_time (epoch seconds)
  - mlflow_predictions: list of objects with fields
      - start_time (epoch seconds; when prediction refers to)
      - avg_qos_0 (numeric prediction)

Usage:
  python plot_experiment_actual_vs_mlflow.py \
    --root /path/to/comparison_of_approaches \
    --approach reactive \
    --job job-0 \
    --experiment validation-proactive-...-created_at_... \
    --target pipelines_status_realtime_pipeline_latency \
    --out_dir result_analysis/actual_vs_pred_plots
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    if 'date' not in df_all.columns:
        raise ValueError("Expected 'date' column in preprocessed CSVs")
    df_all['date'] = pd.to_datetime(df_all['date'])
    return df_all.sort_values('date').reset_index(drop=True)


def _load_experiment_json(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Experiment JSON not found: {path}")
    with open(path, 'r') as f:
        return json.load(f)


def _extract_window_and_predictions(meta: dict) -> ExperimentWindow:
    start_str = meta.get('start_time')
    end_str = meta.get('end_time')
    if start_str is None or end_str is None:
        raise ValueError("Missing start_time/end_time in experiment JSON")
    start_epoch = float(start_str)
    end_epoch = float(end_str)
    return ExperimentWindow(start_epoch_seconds=start_epoch, end_epoch_seconds=end_epoch, source_file=meta.get('source_file', 'experiment.json'))


def _slice_by_window(df: pd.DataFrame, start_epoch: float, end_epoch: float) -> pd.DataFrame:
    start_time = pd.to_datetime(start_epoch, unit='s')
    end_time = pd.to_datetime(end_epoch, unit='s')
    df_exp = df[(df['date'] >= start_time) & (df['date'] <= end_time)].copy()
    return df_exp.sort_values('date').reset_index(drop=True)


def _predictions_from_meta(meta: dict) -> pd.DataFrame:
    preds = meta.get('mlflow_predictions')
    if preds is None:
        raise ValueError("Experiment JSON missing 'mlflow_predictions' list")
    if not isinstance(preds, list):
        raise ValueError("'mlflow_predictions' must be a list of objects")

    rows: List[dict] = []
    for p in preds:
        ts = p.get('start_time')
        val = p.get('avg_qos_0')
        if ts is None or val is None:
            # skip malformed entries
            continue
        rows.append({'date': pd.to_datetime(float(ts), unit='s'), 'avg_qos_0': float(val)})

    if not rows:
        raise ValueError('No valid prediction entries found in mlflow_predictions')
    df = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    return df


def _read_mlflow_csv(csv_path: str) -> pd.DataFrame:
    # Try common separators
    for sep in [';', ',']:
        try:
            df = pd.read_csv(csv_path, sep=sep)
        except Exception:
            continue
        if 'start_time' in df.columns and 'avg_qos_0' in df.columns:
            out = df[['start_time', 'avg_qos_0']].copy()
            out['date'] = pd.to_datetime(out['start_time'].astype(float), unit='s')
            return out[['date', 'avg_qos_0']].sort_values('date').reset_index(drop=True)
    raise ValueError(f"Could not parse MLflow CSV or missing required columns: {csv_path}")


def _find_mlflow_csv_in_dir(directory: str, approach: str, job: str) -> Optional[str]:
    if not os.path.isdir(directory):
        return None
    candidates = [f for f in os.listdir(directory) if f.endswith('.csv') and '_mlflow_results' in f]
    # Heuristics: prefer files containing approach and job in name
    scored: List[tuple] = []
    for name in candidates:
        score = 0
        if approach in name:
            score += 2
        if job in name:
            score += 1
        scored.append((score, name))
    if not scored:
        return None
    scored.sort(reverse=True)
    return os.path.join(directory, scored[0][1])


def _find_mlflow_csv_by_experiment(directory: str, experiment_base: str) -> Optional[str]:
    if not os.path.isdir(directory):
        return None
    candidates = [f for f in os.listdir(directory) if f.endswith('.csv') and '_mlflow_results' in f]
    # Prefer exact startswith base + '_mlflow_results'
    exact = [f for f in candidates if f.startswith(experiment_base + '_mlflow_results')]
    if exact:
        return os.path.join(directory, sorted(exact)[-1])
    # Fallback: contains base anywhere
    contains = [f for f in candidates if experiment_base in f]
    if contains:
        return os.path.join(directory, sorted(contains)[-1])
    return None


def main():
    parser = argparse.ArgumentParser(description='Plot actual metrics vs MLflow predictions for one experiment')
    parser.add_argument('--root', required=True, help='Path to comparison_of_approaches root directory')
    parser.add_argument('--approach', required=True, help='Approach subfolder containing the experiment')
    parser.add_argument('--job', default='job-0', help='Job subdirectory (default: job-0)')
    parser.add_argument('--experiment', required=False, help='Experiment base name without extension (preferred)')
    parser.add_argument('--experiment_json', required=False, help='Experiment JSON filename under raw/ (deprecated)')
    parser.add_argument('--target', required=True, help='Target column name to plot on both series')
    parser.add_argument('--sla', type=float, default=0.2, help='SLA threshold to draw as horizontal line (default: 0.2)')
    parser.add_argument('--mlflow_csv', default=None, help='Path to a MLflow results CSV (fallback if JSON lacks predictions)')
    parser.add_argument('--mlflow_dir', default=None, help='Directory to search for a MLflow results CSV (fallback)')
    parser.add_argument('--out_dir', default='result_analysis/actual_vs_pred_plots', help='Directory to write the plot image')
    args = parser.parse_args()

    approach_path = os.path.join(args.root, args.approach)
    job_path = os.path.join(approach_path, args.job)
    preprocessed_dir = os.path.join(job_path, 'preprocessed')
    raw_dir = os.path.join(job_path, 'raw')
    # Resolve experiment base name and JSON path
    if args.experiment:
        experiment_base = args.experiment
        exp_json_name = experiment_base + '.json'
    elif args.experiment_json:
        exp_json_name = args.experiment_json
        experiment_base = os.path.splitext(exp_json_name)[0]
    else:
        raise ValueError('Please pass --experiment (preferred) or --experiment_json')

    exp_path = os.path.join(raw_dir, exp_json_name)
    if not os.path.isfile(exp_path):
        # Try preprocessed dir as fallback
        alt_path = os.path.join(preprocessed_dir, exp_json_name)
        if os.path.isfile(alt_path):
            exp_path = alt_path
        else:
            raise FileNotFoundError(f"Experiment JSON not found in raw/ or preprocessed/: {exp_json_name}")

    # Load JSON and window
    meta = _load_experiment_json(exp_path)
    meta['source_file'] = os.path.basename(exp_path)
    window = _extract_window_and_predictions(meta)

    # Predictions: from JSON if available, else CSV/dir fallback
    try:
        df_pred = _predictions_from_meta(meta)
    except Exception:
        df_pred = None
    if df_pred is None:
        # Auto-discover MLflow CSV next to JSON first, then in preprocessed
        csv_path: Optional[str] = _find_mlflow_csv_by_experiment(os.path.dirname(exp_path), experiment_base)
        if not csv_path:
            csv_path = _find_mlflow_csv_by_experiment(preprocessed_dir, experiment_base)
        if not csv_path:
            # Last chance: user-provided hints
            if args.mlflow_csv:
                csv_path = args.mlflow_csv
            elif args.mlflow_dir:
                csv_path = _find_mlflow_csv_in_dir(args.mlflow_dir, args.approach, args.job)
        if not csv_path:
            raise ValueError("No MLflow predictions found in JSON or via auto-discovery; provide --mlflow_csv or --mlflow_dir")
        df_pred = _read_mlflow_csv(csv_path)

    # Load actual metrics and slice to window
    df_all = _load_preprocessed_metrics(preprocessed_dir)
    if args.target not in df_all.columns:
        raise ValueError(f"Target column '{args.target}' not found in actuals")
    df_actual = _slice_by_window(df_all, window.start_epoch_seconds, window.end_epoch_seconds)
    if df_actual.empty:
        raise RuntimeError('No actual metrics found for the experiment window')

    # Shift prediction timestamps one step earlier (align to previous index)
    df_pred['date'] = df_pred['date'].shift(1)
    df_pred = df_pred.dropna(subset=['date']).reset_index(drop=True)

    # Clip predictions to window bounds after shifting
    df_pred = _slice_by_window(df_pred, window.start_epoch_seconds, window.end_epoch_seconds)
    if df_pred.empty:
        raise RuntimeError('No MLflow predictions fall within the experiment window')

    # Limit to first 2 hours of the experiment
    plot_start = pd.to_datetime(window.start_epoch_seconds, unit='s')
    plot_end = plot_start + pd.Timedelta(hours=2)
    df_actual = df_actual[df_actual['date'] <= plot_end].reset_index(drop=True)
    df_pred = df_pred[df_pred['date'] <= plot_end].reset_index(drop=True)

    # Plot together
    plt.figure(figsize=(14, 6))
    # Color Actual by predicted decision: Edge if predicted <= SLA, Cloud if > SLA
    dates_actual = df_actual['date'].sort_values().to_numpy()
    y_actual = df_actual[args.target].astype(float).to_numpy()
    aligned = pd.merge_asof(
        pd.DataFrame({'date': df_actual['date'].sort_values()}),
        df_pred[['date', 'avg_qos_0']].sort_values('date'),
        on='date',
        direction='backward',
    )
    pred_vals = aligned['avg_qos_0'].astype(float)
    pred_vals = pred_vals.fillna(method='ffill')
    pred_vals = pred_vals.fillna(0.0)
    is_cloud = pred_vals > float(args.sla)

    # Segment actual series where predicted decision changes
    run_start = 0
    first_edge = True
    first_cloud = True
    for i in range(1, len(dates_actual) + 1):
        end_run = (i == len(dates_actual)) or (is_cloud.iloc[i] != is_cloud.iloc[i - 1])
        if end_run:
            color = 'C1' if is_cloud.iloc[run_start] else 'C0'
            label = None
            if is_cloud.iloc[run_start] and first_cloud:
                label = 'Actual (Cloud by predicted)'
                first_cloud = False
            if not is_cloud.iloc[run_start] and first_edge:
                label = 'Actual (Edge by predicted)'
                first_edge = False
            plt.plot(dates_actual[run_start:i], y_actual[run_start:i], color=color, linewidth=2, label=label)
            run_start = i

    # Plot prediction single-color
    plt.plot(df_pred['date'], df_pred['avg_qos_0'].astype(float), label='Prediction', color='C2', linewidth=2, alpha=0.9)
    plt.axhline(y=args.sla, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='SLA')
    plt.title('Actual vs Prediction')
    plt.xlabel('Time')
    plt.ylabel('Latency')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    safe_exp = os.path.splitext(window.source_file)[0]
    out_path = os.path.join(args.out_dir, f"actual_vs_mlflow__{args.approach}__{args.job}__{safe_exp}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot: {out_path}")


if __name__ == '__main__':
    main()


