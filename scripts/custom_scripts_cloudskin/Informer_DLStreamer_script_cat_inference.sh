export CUDA_VISIBLE_DEVICES=0

project_name=DLStreamer
model_name=Informer
seq_len=10
pred_len=10
label_len=10
enc_size=10

# Create necessary directories
mkdir -p ./learning-plane/logs
mkdir -p ./learning-plane/logs/$project_name
mkdir -p ./learning-plane/logs/$project_name/$model_name

python -u ./libs/TSLib/run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./learning-plane/data/datascience_training \
  --data_path preprocessed_data.csv \
  --model_id custom \
  --model $model_name \
  --data custom \
  --features MS \
  --target pipelines_status_realtime_pipeline_latency \
  --categorical_cols cluster \
  --use_gpu False \
  --freq t \
  --inverse True \
  --pred_len $pred_len \
  --seq_len $seq_len \
  --label_len $label_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $enc_size \
  --dec_in $enc_size \
  --c_out 1 \
  --train_epochs 10 \
  --patience 5 \
  --batch_size 32 \
  --d_model 256 \
  --d_ff 512 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'MSE'  | tee ./learning-plane/logs/$project_name/$model_name/$model_name'_'$seq_len'_pl'$pred_len'_enc'$enc_size'.log'
