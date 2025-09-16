from __future__ import annotations

import os
import numpy as np
import joblib
from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_RF_Pred


class Exp_Random_Forest:
    """
    Inference experiment for a trained Random Forest regressor.
    - No scaling. One-hot binary feature for the specified cluster ID.
    - Builds a single-sample feature vector from the last row and tiles prediction across pred_len.
    """

    FEATURE_ORDER = [
        "num__node_cpu_usage",
        "num__node_mem_usage",
        "num__number_pipelines",
        "num__pipelines_server_cpu_usage",
        "num__pipelines_server_mem_usage",
        "num__quarter_15m",
        "num__hour",
        "cat__cluster_fd7816db-7948-4602-af7a-1d51900792a7",
    ]

    def __init__(self, args):
        self.args = args
        model_path = os.path.join(self.args.checkpoints, self.args.model_id, "model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Random Forest checkpoint not found at: {model_path}")
        self.model = joblib.load(model_path)

    def predict(self, setting: str, load: bool = False):
        ds = Dataset_RF_Pred(self.args, self.args.root_path, self.args.data_path, feature_order=self.FEATURE_ORDER)
        loader = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

        for vec_batch in loader:
            vec = vec_batch[0].numpy().reshape(1, -1)
            y_hat = float(self.model.predict(vec)[0])
            break

        pred = np.full((1, self.args.pred_len, 1), y_hat, dtype=float)
        return pred