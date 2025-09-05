#!/usr/bin/env python3
"""
Compare approaches using aggregated outputs produced by aggregate_experiments_by_time_delta.py.

It reads each approach's aggregated CSV and creates:
  - A summary CSV with approach-level stats (overall mean of means, mean of maxes, peak max, etc.)
  - An optional line plot comparing per-step mean and max across approaches.

Usage:
  python compare_approaches_from_aggregates.py \
    --root /home/jolivera/Documents/CloudSkin/Time-Series-Library/dataset/comparison_of_approaches \
    --job job-0 \
    [--output_subdir aggregated] \
    [--summary_out result_analysis/approach_comparison_summary.csv] \
    [--plot_out result_analysis/approach_comparison_plot.png]
"""

import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def _discover_approaches(root: str) -> List[str]:
    return [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]


def _load_aggregate_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {'step_index', 'time_since_start_seconds', 'mean_over_experiments', 'max_over_experiments'}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns in aggregate CSV {path}: {missing}")
    return df.sort_values('step_index').reset_index(drop=True)


def summarize_approach(df: pd.DataFrame, sla: float) -> Dict[str, float]:
    """Summarize approach-level metrics including SLA-based features.

    - overall_mean_of_means: Average of mean_over_experiments over all steps
    - overall_mean_of_maxes: Average of max_over_experiments over all steps
    - peak_max: Maximum of max_over_experiments over all steps
    - num_times_crossed_above_sla: Count of rising-edge crossings where mean goes from < SLA to >= SLA
    - total_seconds_above_sla: Sum of time intervals where mean >= SLA
    """
    overall_mean_of_means = float(df['mean_over_experiments'].mean())
    overall_mean_of_maxes = float(df['max_over_experiments'].mean())
    peak_max = float(df['max_over_experiments'].max())

    above = df['mean_over_experiments'] >= sla
    # Count rising edges: from below (False) to True
    crossings = int((above.astype(int).diff().fillna(0) == 1).sum())

    # Time deltas to next step
    dt_next = df['time_since_start_seconds'].shift(-1) - df['time_since_start_seconds']
    dt_next = dt_next.fillna(0)
    total_seconds_above_sla = float((above.astype(float) * dt_next).sum())

    return {
        'num_steps': int(df.shape[0]),
        'overall_mean_of_means': overall_mean_of_means,
        'overall_mean_of_maxes': overall_mean_of_maxes,
        'peak_max': peak_max,
        'num_times_crossed_above_sla': crossings,
        'total_seconds_above_sla': total_seconds_above_sla,
    }


def main():
    parser = argparse.ArgumentParser(description='Compare approaches from aggregated experiment results')
    parser.add_argument('--root', required=True, help='Path to comparison_of_approaches root directory')
    parser.add_argument('--job', default='job-0', help='Job subdirectory to process (default: job-0)')
    parser.add_argument('--output_subdir', default='aggregated', help='Subdirectory under job containing aggregated CSVs')
    parser.add_argument('--summary_out', default=None, help='Where to write the summary CSV')
    parser.add_argument('--plot_out', default=None, help='Where to write a comparison plot (optional)')
    parser.add_argument('--sla', type=float, default=0.2, help='SLA threshold for latency; default 0.2')
    args = parser.parse_args()

    root = args.root
    if not os.path.isdir(root):
        raise NotADirectoryError(f"Root not found: {root}")

    approaches = _discover_approaches(root)
    summaries: List[Dict[str, float]] = []
    approach_to_df: Dict[str, pd.DataFrame] = {}

    for approach in sorted(approaches):
        agg_csv = os.path.join(root, approach, args.job, args.output_subdir, 'aggregated_realtime_latency_by_time_delta.csv')
        if not os.path.isfile(agg_csv):
            # Skip approaches without aggregates
            continue
        df = _load_aggregate_csv(agg_csv)
        summary = summarize_approach(df, sla=args.sla)
        summary['approach'] = approach
        summaries.append(summary)
        approach_to_df[approach] = df

    if not summaries:
        raise RuntimeError('No approaches with aggregated CSVs found. Run aggregate script first.')

    df_summary = pd.DataFrame(summaries).sort_values('overall_mean_of_means').reset_index(drop=True)

    if args.summary_out:
        os.makedirs(os.path.dirname(args.summary_out), exist_ok=True)
        df_summary.to_csv(args.summary_out, index=False,sep=';')
        print(f"Summary written to {args.summary_out}")
    else:
        print(df_summary)

    if args.plot_out:
        # Plot mean and max per approach
        plt.figure(figsize=(14, 8))
        for approach, df in approach_to_df.items():
            plt.plot(df['time_since_start_seconds'] / 60.0, df['mean_over_experiments'], label=f"{approach} - mean")
        plt.title('Mean realtime latency over experiment (aligned by time delta)')
        plt.xlabel('Minutes since experiment start')
        plt.ylabel('Mean realtime latency')
        plt.grid(True, alpha=0.3)
        plt.legend()
        os.makedirs(os.path.dirname(args.plot_out), exist_ok=True)
        plt.tight_layout()
        plt.savefig(args.plot_out, dpi=200)
        print(f"Mean comparison plot saved to {args.plot_out}")

        # Second figure for max
        plt.figure(figsize=(14, 8))
        for approach, df in approach_to_df.items():
            plt.plot(df['time_since_start_seconds'] / 60.0, df['max_over_experiments'], label=f"{approach} - max")
        plt.title('Max realtime latency over experiment (aligned by time delta)')
        plt.xlabel('Minutes since experiment start')
        plt.ylabel('Max realtime latency')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        base, ext = os.path.splitext(args.plot_out)
        plot2 = f"{base}_max{ext or '.png'}"
        plt.savefig(plot2, dpi=200)
        print(f"Max comparison plot saved to {plot2}")


if __name__ == '__main__':
    main()


