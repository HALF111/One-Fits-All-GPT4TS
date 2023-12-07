export CUDA_VISIBLE_DEVICES=0

seq_len=104
model=GPT4TS

gpt_layer=6


test_train_num=200
selected_data_num=3
adapted_lr_times=0.001


# for pred_len in 24 36 48 60
for pred_len in 24
do
for percent in 100
do
echo $random_seed

# python main.py \
python main_test.py \
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
    --run_select_with_distance \
    --test_train_num $test_train_num \
    --selected_data_num $selected_data_num \
    --adapted_lr_times $adapted_lr_times \
    --adapt_cycle \
    --train_epochs 0
done
done

