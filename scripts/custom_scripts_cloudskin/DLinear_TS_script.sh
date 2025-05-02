export CUDA_VISIBLE_DEVICES=0
#GATE D6
model_name=DLinear
seq_len=10
pred_len=3
enc_size=10

if [ ! -d "./logs/$model_name" ]; then
    mkdir ./logs/$model_name
fi

python -u ./Time-Series-Library/run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./load-data \
  --data_path final_df.csv \
  --model_id custom \
  --model $model_name \
  --data custom \
  --features MS \
  --target PredictionTimeTS \
  --inverse True \
  --pred_len $pred_len \
  --seq_len $seq_len \
  --label_len 3 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $enc_size \
  --dec_in $enc_size \
  --c_out 1 \
  --train_epochs 10 \
  --patience 2 \
  --batch_size 32 \
  --d_model 32 \
  --d_ff 128 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'MSE'  | tee logs/$model_name/$model_name'_'$seq_len'_pl'$pred_len'_enc'$enc_size'.log'

model_name=Mamba
if [ ! -d "./logs/$model_name" ]; then
    mkdir ./logs/$model_name
fi

python -u ./Time-Series-Library/run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./load-data \
  --data_path final_df.csv \
  --model_id custom \
  --model $model_name \
  --data custom \
  --features MS \
  --target PredictionTimeTS \
  --categorical_cols node,hour \
  --inverse True \
  --pred_len $pred_len \
  --seq_len $seq_len \
  --label_len 3 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in $enc_size \
  --expand 2 \
  --c_out 1 \
  --train_epochs 10 \
  --patience 2 \
  --batch_size 32 \
  --d_conv 4 \
  --d_ff 16 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'MSE'  | tee logs/$model_name/$model_name'_'$seq_len'_pl'$pred_len'_enc'$enc_size'.log'


model_name=Nonstationary_Transformer
if [ ! -d "./logs/$model_name" ]; then
    mkdir ./logs/$model_name
fi

python -u ./Time-Series-Library/run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./load-data \
  --data_path final_df.csv \
  --model_id custom \
  --model $model_name \
  --data custom \
  --features MS \
  --target PredictionTimeTS \
  --categorical_cols node,hour \
  --inverse True \
  --pred_len $pred_len \
  --seq_len $seq_len \
  --label_len 3 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $enc_size \
  --dec_in $enc_size \
  --c_out 1 \
  --train_epochs 10 \
  --patience 2 \
  --batch_size 32 \
  --d_model 32 \
  --d_ff 128 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'MSE'  | tee logs/$model_name/$model_name'_'$seq_len'_pl'$pred_len'_enc'$enc_size'.log'
