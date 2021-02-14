cd ..

# 538 tasks in total

CUDA_VISIBLE_DEVICES=9 \
python cli_grouped.py \
 --checkpoint /home/qinyuan/zs/out/bart-base-squad-LR1e-5-EPO50/best-model.pt \
 --adapter_dim 8 \
 --seed 55 \
 --dataset squad_grouped \
 --inner_bsz 16 \
 --do_train \
 --do_predict \
 --freeze_embeds \
 --total_steps 1200 \
 --warmup_steps 72 \
 --max_grad_norm 0.1 \
 --weight_decay 0.01 \
 --model facebook/bart-base \
 --output_dir out/bart-base-squad-grouped-test  \
 --train_file data/squad/zs_train.json \
 --predict_file data/squad/zs_dev.json \
 --train_batch_size 1 \
 --gradient_accumulation_steps 4 \
 --predict_batch_size 16 \
 --max_input_length 512 \
 --max_output_length 64 \
 --eval_period 80 \
 --num_train_epochs 60;
 

