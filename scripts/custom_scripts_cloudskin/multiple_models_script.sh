export CUDA_VISIBLE_DEVICES=0
#GATE D6
model_name=Informer
seq_len=10
pred_len=3
enc_size=10
if [ ! -d "./logs/$model_name" ]; then
    mkdir ./logs/$model_name
fi

for model_name in Informer
do
    for seq_len in 5 10 15 20
    do
        for pred_len in 3 5 10
        do
            python -u ./TSLib/run.py \
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
            --loss 'MSE' | tee logs/$model_name/$model_name'_'$seq_len'_pl'$pred_len'_enc'$enc_size'.log'
        done
    done
done
