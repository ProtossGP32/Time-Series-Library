#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallel MLP GridSearch Training Script
Author: Joan Oliveras
Purpose: Train and evaluate MLPRegressor hyperparameters in parallel.
"""

import pandas as pd
import numpy as np
import joblib
import time
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ======================================
# Data loading and splitting
# ======================================
def split_by_app_user(df, train_users, val_users, test_users):
    train_df = df[df["torchserve_app_user"].isin(train_users)].copy()
    val_df   = df[df["torchserve_app_user"].isin(val_users)].copy()
    test_df  = df[df["torchserve_app_user"].isin(test_users)].copy()
    return train_df, val_df, test_df


def calculate_split_percentages(train_df, val_df, test_df):
    total = len(train_df) + len(val_df) + len(test_df)
    return {
        "train_pct": len(train_df) / total * 100,
        "val_pct": len(val_df) / total * 100,
        "test_pct": len(test_df) / total * 100,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "total_rows": total
    }


def make_xy(df):
    X = df[[
        "torchserve_app_user",
        "node_id_src",
        "node_id_tgt",
        "torchserve_node_cpu_src",
        "torchserve_node_energy_src",
        "torchserve_node_power_src",
        "torchserve_app_cpu_src",
        "torchserve_app_energy_src",
        "torchserve_app_power_src",
        "torchserve_app_latency_src",
        "torchserve_app_qps_src"
    ]].copy()

    y = df[[
        "torchserve_node_cpu_tgt",
        "torchserve_node_energy_tgt",
        "torchserve_node_power_tgt",
        "torchserve_app_cpu_tgt",
        "torchserve_app_energy_tgt",
        "torchserve_app_power_tgt",
        "torchserve_app_latency_tgt",
        "torchserve_app_qps_tgt"
    ]].copy()
    return X, y


def shuffle_df(X, y, random_state=42):
    shuffled_idx = X.sample(frac=1, random_state=random_state).index
    return X.loc[shuffled_idx].reset_index(drop=True), y.loc[shuffled_idx].reset_index(drop=True)


def evaluate_model(model, X_val, y_val):
    """Evaluate model performance and return a summary DataFrame."""
    targets = list(y_val.columns)
    y_pred = model.predict(X_val)

    if isinstance(y_pred, list):
        y_pred = np.column_stack(y_pred)

    results, r2_list, mae_list, rmse_list, mape_list = [], [], [], [], []

    for i, target in enumerate(targets):
        y_true_t = y_val[target].values
        y_pred_t = y_pred[:, i]

        r2 = r2_score(y_true_t, y_pred_t)
        mae = mean_absolute_error(y_true_t, y_pred_t)
        rmse = mean_squared_error(y_true_t, y_pred_t, squared=False)
        mape = np.mean(np.abs((y_true_t - y_pred_t) / np.clip(y_true_t, 1e-8, None))) * 100

        results.append([target, r2, mae, rmse, mape])
        r2_list.append(r2)
        mae_list.append(mae)
        rmse_list.append(rmse)
        mape_list.append(mape)

    overall_r2_mean = np.mean(r2_list)
    overall_r2_global = r2_score(y_val, y_pred)
    overall_mae = np.mean(mae_list)
    overall_rmse = np.mean(rmse_list)
    overall_mape = np.mean(mape_list)

    results_df = pd.DataFrame(results, columns=["Target", "R2", "MAE", "RMSE", "MAPE"])
    overall_row = pd.DataFrame({
        "Target": ["Overall (mean)", "Overall (global)"],
        "R2": [overall_r2_mean, overall_r2_global],
        "MAE": [overall_mae, None],
        "RMSE": [overall_rmse, None],
        "MAPE": [overall_mape, None]
    })
    results_df = pd.concat([results_df, overall_row], ignore_index=True)

    return results_df


# ======================================
# Main training logic
# ======================================
def main():
    start_time = time.time()

    df = pd.read_csv(
        "/home/jolivera/Documents/CloudSkin/Time-Series-Library/ccgrid_scripts/data/training_data_scored_pairs.csv"
    )

    train_users = [1, 13, 25, 31, 43, 55]
    val_users   = [7, 19]
    test_users  = [37, 49]

    train_df, val_df, test_df = split_by_app_user(df, train_users, val_users, test_users)
    print("Data split:", calculate_split_percentages(train_df, val_df, test_df))

    X_train, y_train = make_xy(train_df)
    X_val, y_val = make_xy(val_df)
    X_test, y_test = make_xy(test_df)

    X_train, y_train = shuffle_df(X_train, y_train)
    X_val, y_val = shuffle_df(X_val, y_val)

    cat_features = ["node_id_src", "node_id_tgt"]
    num_features = [col for col in X_train.columns if col not in cat_features]

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[cat_features])

    def encode_transform(X):
        X_cat = pd.DataFrame(
            encoder.transform(X[cat_features]),
            columns=encoder.get_feature_names_out(cat_features),
            index=X.index
        )
        return X_cat

    x_scaler = MinMaxScaler()
    x_scaler.fit(X_train[num_features])

    def scale_transform(X):
        X_num = pd.DataFrame(
            x_scaler.transform(X[num_features]),
            columns=num_features,
            index=X.index
        )
        return X_num

    def prepare(X):
        return pd.concat([scale_transform(X), encode_transform(X)], axis=1)

    X_train_prepared = prepare(X_train)
    X_val_prepared = prepare(X_val)
    X_test_prepared = prepare(X_test)

    # =======================
    # Grid Search
    # =======================
    param_grid = {
        "hidden_layer_sizes": [(128,64,32), (256,128,64), (128,128,64,32)],
        "activation": ["relu", "tanh"],
        "learning_rate_init": [0.001, 0.003, 0.005],
        "alpha": [0.0001, 0.001, 0.01],
    }

    mlp_base = MLPRegressor(
        max_iter=1000,
        early_stopping=True,
        random_state=42
    )

    grid = GridSearchCV(
        mlp_base,
        param_grid,
        cv=3,
        scoring="r2",
        n_jobs=-1,
        verbose=2
    )

    print("\nStarting grid search...")
    grid.fit(X_train_prepared, y_train)
    print("\nBest parameters:", grid.best_params_)
    print("Best CV R2:", grid.best_score_)

    # Save results and best model
    os.makedirs("mlp_results", exist_ok=True)
    joblib.dump(grid.best_estimator_, "mlp_results/best_mlp_model.pkl")
    pd.DataFrame(grid.cv_results_).to_csv("mlp_results/gridsearch_results.csv", index=False)

    # Evaluate on validation set
    print("\nEvaluating best model...")
    results_df = evaluate_model(grid.best_estimator_, X_val_prepared, y_val)
    results_df.to_csv("mlp_results/validation_results.csv", index=False)

    print("\nValidation results:")
    print(results_df)

    print(f"\nTotal runtime: {(time.time() - start_time):.2f} seconds")


# ======================================
# Run
# ======================================
if __name__ == "__main__":
    main()
