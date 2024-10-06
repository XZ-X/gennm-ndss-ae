#!/bin/bash -x


WANDB_ENTITY=<WANDB-NAME> \
WANDB_PROJECT=<WANDB-PROJECT> \
torchrun --nproc_per_node=2 --master_port 14285 \
    src/train.py \
    --stage sft \
    --model_name_or_path google/codegemma-2b \
    --run_name gennm \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --val_size 0.01 \
    --cutoff_len 2048 \
    --dataset gennm-dirty \
    --template empty \
    --train_on_prompt \
    --finetuning_type full \
    --preprocessing_num_workers 16 \
    --output_dir ckpt-gennm \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --per_device_eval_batch_size 2 \
    --lr_scheduler_type cosine \
    --warmup_steps 2000 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --num_train_epochs 4.0 \
    --plot_loss \
    --bf16 \
    --deepspeed deep_speed_config.json \
    --hub_strategy all_checkpoints