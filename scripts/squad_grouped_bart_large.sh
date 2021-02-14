cd ..

CUDA_VISIBLE_DEVICES=0 \
python cli_grouped.py \
 --checkpoint /home/$USER/zs/out/bart-large-squad-LR1e-5-EPO20/best-model.pt \
 --adapter_dim 8 \
 --seed 55 \
 --dataset squad_grouped \
 --inner_bsz 16 \
 --do_predict \
 --freeze_embeds \
 --total_steps 1200 \
 --warmup_steps 72 \
 --max_grad_norm 0.1 \
 --weight_decay 0.01 \
 --model facebook/bart-large \
 --output_dir out/bart-large-squad-grouped-test  \
 --train_file data/squad/zs_train.json \
 --predict_file data/squad/zs_test.json \
 --train_batch_size 1 \
 --gradient_accumulation_steps 4 \
 --predict_batch_size 16 \
 --max_input_length 512 \
 --max_output_length 64 \
 --eval_period 80 \
 --num_train_epochs 60;
 

