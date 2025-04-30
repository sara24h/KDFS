#!/bin/bash

# Model and dataset configurations
arch=ResNet_50
result_dir=/kaggle/working/results/run_resnet50_imagenet_prune1
dataset_dir=/kaggle/input/rvf10k
dataset_mode=rvf10k
rvf10k_train_csv=/kaggle/input/rvf10k/train.csv
rvf10k_valid_csv=/kaggle/input/rvf10k/valid.csv
teacher_ckpt_path=/kaggle/working/KDFS/teacher_dir/teacher_model.pth
device=cuda


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUBLAS_WORKSPACE_CONFIG=:4096:8


if [ ! -f "$teacher_ckpt_path" ]; then
    echo "Error: Teacher checkpoint not found at $teacher_ckpt_path"
    exit 1
fi
if [ ! -f "$rvf10k_train_csv" ]; then
    echo "Error: Train CSV file not found at $rvf10k_train_csv"
    exit 1
fi
if [ ! -f "$rvf10k_valid_csv" ]; then
    echo "Error: Valid CSV file not found at $rvf10k_valid_csv"
    exit 1
fi
if [ ! -d "$dataset_dir" ]; then
    echo "Error: Dataset directory not found at $dataset_dir"
    exit 1
fi


mkdir -p $result_dir

# Run training
python /kaggle/working/KDFS/main.py \
    --phase train \
    --dataset_mode $dataset_mode \
    --dataset_dir $dataset_dir \
    --rvf10k_train_csv $rvf10k_train_csv \
    --rvf10k_valid_csv $rvf10k_valid_csv \
    --num_workers 4 \
    --pin_memory \
    --device $device \
    --arch $arch \
    --seed 3407 \
    --result_dir $result_dir \
    --teacher_ckpt_path $teacher_ckpt_path \
    --num_epochs 100 \
    --lr 4e-3 \
    --warmup_steps 30 \
    --warmup_start_lr 4e-5 \
    --lr_decay_T_max 250 \
    --lr_decay_eta_min 4e-5 \
    --weight_decay 2e-5 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --target_temperature 3 \
    --gumbel_start_temperature 1 \
    --gumbel_end_temperature 0.1 \
    --coef_kdloss 0.05 \
    --coef_rcloss 1000 \
    --coef_maskloss 10000 \
    --compress_rate 0.6 \
&& \
# Run finetuning
python /kaggle/working/KDFS/main.py \
    --phase finetune \
    --dataset_mode $dataset_mode \
    --dataset_dir $dataset_dir \
    --rvf10k_train_csv $rvf10k_train_csv \
    --rvf10k_valid_csv $rvf10k_valid_csv \
    --num_workers 4 \
    --pin_memory \
    --device $device \
    --arch $arch \
    --seed 3407 \
    --result_dir $result_dir \
    --finetune_student_ckpt_path $result_dir"/student_model/"$arch"_sparse_last.pt" \
    --finetune_num_epochs 6 \
    --finetune_lr 4e-6 \
    --finetune_warmup_steps 5 \
    --finetune_warmup_start_lr 4e-8 \
    --finetune_lr_decay_T_max 20 \
    --finetune_lr_decay_eta_min 4e-8 \
    --finetune_weight_decay 2e-5 \
    --finetune_train_batch_size 8 \
    --finetune_eval_batch_size 8 \
    --sparsed_student_ckpt_path $result_dir"/student_model/finetune_"$arch"_sparse_best.pt"
