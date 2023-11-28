
export CUDA_VISIBLE_DEVICES=1

seq_len=336
model=GPT4TS

gpt_layer=6

seed_lst=$(seq 50)

for percent in 100
do
# for pred_len in 96 192 336 720
for pred_len in 96
do
for lr in 0.0001
do
for random_seed in $seed_lst
do
echo $random_seed

python main.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data ett_h \
    --seq_len $seq_len \
    --label_len 168 \
    --pred_len $pred_len \
    --batch_size 256 \
    --lradj type4 \
    --learning_rate $lr \
    --train_epochs 10 \
    --decay_fac 0.5 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer $gpt_layer \
    --itr 1 \
    --model $model \
    --tmax 20 \
    --cos 1 \
    --is_gpt 1 \
    --random_seed $random_seed

done
done
done
done
