model_name=RandomForest
seq_len=10
pred_len=10

python -u run.py \
  --output_path /home/jolivera/Documents/CloudSkin/Time-Series-Library/dataset/partial_validation_dataset_proactive_linear_regression/5_min_results_random_forest \
  --task_name random_forest \
  --is_training 0 \
  --root_path /home/jolivera/Documents/CloudSkin/Time-Series-Library/dataset/partial_validation_dataset_proactive_linear_regression/5_min_intervals \
  --data_path preprocessed_data.csv \
  --data_iterate True \
  --model_id random_forest_n_estimators_100_max_depth_10 \
  --model $model_name \
  --data random_forest \
  --target pipelines_status_realtime_pipeline_latency \
  --pred_len $pred_len \
  --seq_len $seq_len \
  --c_out 1