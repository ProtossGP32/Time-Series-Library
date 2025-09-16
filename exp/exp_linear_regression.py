from __future__ import annotations

import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_LR_Pred


class Exp_Linear_Regression:
    """
    Lightweight production inference experiment for a fixed-coefficient
    2 Linear Regression model with two variants: edge and cloud. The
    dataset aggregates the last `seq_len` rows into a single feature
    vector using the following features and also yields a flag indicating
    whether the cluster is edge (1) or cloud (0):
      - num__node_mem_usage
      - num__number_pipelines
      - num__node_cpu_usage
      - node_cpu_x_server_cpu
    A linear combination (no intercept) is applied using the respective
    coefficients, and the scalar is repeated across `pred_len` steps.
    """

    # Fixed order must match Dataset_LR_Pred.FEATURE_ORDER
    FEATURE_ORDER = [
        "num__node_mem_usage",
        "num__number_pipelines",
        "num__node_cpu_usage",
        "node_cpu_x_server_cpu",
    ]

    # Coefficients learned offline (no intercept; model without constant)
    WEIGHTS_EDGE = {
        "num__node_mem_usage": 2.369795e-11,
        "num__number_pipelines": 2.421762e-01,
        "num__node_cpu_usage": -1.535301e-01,
        "node_cpu_x_server_cpu": 6.267838e-03,
    }

    WEIGHTS_CLOUD = {
        "num__node_mem_usage": 5.512106e-12,
        "num__number_pipelines": -8.996437e-03,
        "num__node_cpu_usage": 2.404007e-03,
        "node_cpu_x_server_cpu": 7.405377e-04,
    }

    # Required base columns in the CSV
    REQUIRED_COLUMNS = [
        "date",
        "cluster",
        "number_pipelines",
        "node_mem_usage",
        "node_cpu_usage",
        "pipelines_server_cpu_usage",
    ]

    def __init__(self, args):
        self.args = args

    # All data preparation happens in data_loader.Dataset_LR_Pred

    def _predict_single(self, feats: dict, is_edge: bool) -> float:
        # Linear combination, no intercept
        weights = self.WEIGHTS_EDGE if is_edge else self.WEIGHTS_CLOUD
        y = 0.0
        for name, weight in weights.items():
            y += weight * float(feats.get(name, 0.0))
        # Cap to minimum of 0
        return float(max(y, 0.0))

    def predict(self, setting: str, load: bool = False):
        """
        Returns np.ndarray of shape (1, pred_len, 1) to align with downstream expectations.
        Uses a DataLoader over a single-sample dataset for consistency with other exps.
        """
        ds = Dataset_LR_Pred(self.args, self.args.root_path, self.args.data_path)
        loader = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
        for vec_batch, is_edge_batch in loader:
            # vec_batch shape: (1, feature_dim); is_edge_batch shape: (1,)
            vec = vec_batch[0]
            is_edge = bool(is_edge_batch[0].item())
            feats = {name: float(vec[idx].item()) for idx, name in enumerate(self.FEATURE_ORDER)}
            y_hat = self._predict_single(feats, is_edge)
            break

        pred = np.full((1, self.args.pred_len, 1), y_hat, dtype=float)
        return pred


