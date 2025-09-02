from __future__ import annotations

import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_LR_Pred


class Exp_Linear_Regression:
    """
    Lightweight production inference experiment for a fixed-coefficient
    Linear Regression model. Aggregates the last `seq_len` rows into a
    single feature vector by:
      - Averaging numerical columns
      - Taking the last value for categorical columns
    Then constructs interaction features and predicts a single scalar,
    repeated across `pred_len` steps.
    """

    # Coefficients learned offline (no intercept; model without constant)
    WEIGHTS = {
        "node_mem_usage": 4.975791e-12,
        "number_pipelines": -7.332159e-02,
        "cluster_x_pipelines": -5.780488e-03,
        "cluster_x_node_mem": 1.159748e-11,
        "node_cpu_x_server_cpu": 4.546697e-03,
    }

    # Cluster ID used in one-hot encoding during training (only this OHE column kept)
    CLUSTER_REF_OHE_COL = "cat__cluster_fd7816db-7948-4602-af7a-1d51900792a7"
    CLUSTER_REF_VALUE = "fd7816db-7948-4602-af7a-1d51900792a7"

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

    def _predict_single(self, feats: dict) -> float:
        # Linear combination, no intercept
        y = 0.0
        for name, weight in self.WEIGHTS.items():
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
        for vec in loader:
            feats = {
                "node_mem_usage": float(vec[0, 0].item()),
                "number_pipelines": float(vec[0, 1].item()),
                "cluster_x_pipelines": float(vec[0, 2].item()),
                "cluster_x_node_mem": float(vec[0, 3].item()),
                "node_cpu_x_server_cpu": float(vec[0, 4].item()),
            }
            y_hat = self._predict_single(feats)
            break

        pred = np.full((1, self.args.pred_len, 1), y_hat, dtype=float)
        return pred


