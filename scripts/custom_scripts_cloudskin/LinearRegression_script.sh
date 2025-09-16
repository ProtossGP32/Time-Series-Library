model_name=LinearRegression
seq_len=10
pred_len=10

python -u run.py \
  --output_path /home/jolivera/Documents/CloudSkin/Time-Series-Library/dataset/partial_validation_dataset_proactive_linear_regression/5_min_results_linear_regression \
  --task_name linear_regression \
  --is_training 0 \
  --root_path /home/jolivera/Documents/CloudSkin/Time-Series-Library/dataset/partial_validation_dataset_proactive_linear_regression/5_min_intervals \
  --data_path preprocessed_data.csv \
  --data_iterate True \
  --model_id custom \
  --model $model_name \
  --data linear_regression \
  --target pipelines_status_realtime_pipeline_latency \
  --categorical_cols cluster \
  --pred_len $pred_len \
  --seq_len $seq_len \
  --c_out 1 \




