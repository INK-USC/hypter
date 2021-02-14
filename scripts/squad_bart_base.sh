cd ..


LR=3e-5
FREQ=4
GPU=1

echo "FREQ=$FREQ"

CUDA_VISIBLE_DEVICES=${GPU} \
python cli.py \
 --learning_rate $LR \
 --do_train \
 --do_predict \
 --model facebook/bart-base \
 --output_dir out/bart-base-squad-test  \
 --dataset squad \
 --train_file data/squad/zs_train.json \
 --predict_file data/squad/zs_dev.json \
 --max_grad_norm 0.1 \
 --train_batch_size 8 \
 --gradient_accumulation_steps $FREQ \
 --predict_batch_size 16 \
 --max_input_length 512 \
 --max_output_length 64 \
 --eval_period 640 \
 --num_train_epochs 30;
 
echo "FREQ=$FREQ"


