export CUDA_VISIBLE_DEVICES=0

seq_len=104
model=GPT4TS

gpt_layer=6

seed_lst=$(seq 100)

# for pred_len in 24 36 48 60
for pred_len in 24
do
for percent in 100
do
for random_seed in $seed_lst
do
echo $random_seed

python main.py \
    --root_path ./datasets/illness/ \
    --data_path national_illness.csv \
    --model_id illness_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len 18 \
    --pred_len $pred_len \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --freq 0 \
    --enc_in 7 \
    --c_out 7 \
    --patch_size 24 \
    --stride 2 \
    --percent $percent \
    --gpt_layer $gpt_layer \
    --itr 1 \
    --model $model \
    --is_gpt 1 \
    --random_seed $random_seed
done
done
done

