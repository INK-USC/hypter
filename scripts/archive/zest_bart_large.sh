# 2020-12-9

cd /home/qinyuan/zs

CUDA_VISIBLE_DEVICES=8 \
python cli.py \
 --seed 55 \
 --do_train \
 --do_predict \
 --freeze_embeds \
 --total_steps 5050 \
 --warmup_steps 100 \
 --max_grad_norm 0.1 \
 --weight_decay 0.01 \
 --model facebook/bart-large \
 --output_dir out/bart-large-zest55-reproduce  \
 --dataset zest \
 --train_file data/zest_train.jsonl \
 --predict_file data/zest_dev.jsonl \
 --train_batch_size 16 \
 --gradient_accumulation_steps 2 \
 --predict_batch_size 32 \
 --max_input_length 512 \
 --max_output_length 64 \
 --eval_period 10000000 \
 --num_train_epochs 15
 
cd /home/qinyuan/zest/bin
python evaluate-zest.py -p ~/zs/out/bart-large-zest55-reproduce/predictions.txt -d ~/zest/data/dev.jsonl -o ~/zs/out/bart-large-zest55-reproduce/results.json
