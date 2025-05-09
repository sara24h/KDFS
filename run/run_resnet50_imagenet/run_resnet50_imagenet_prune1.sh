#!/bin/bash

# Default values
arch=ResNet_50
result_dir=/kaggle/working/results/run_resnet50_imagenet_prune1
device=cuda
teacher_ckpt_path_default=/kaggle/working/KDFS/teacher_dir/teacher_model_best.pth

# Parse command-line arguments for teacher_ckpt_path
while [ $# -gt 0 ]; do
    case "$1" in
        --teacher_ckpt_path)
            teacher_ckpt_path="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Use default teacher_ckpt_path if not provided
teacher_ckpt_path=${teacher_ckpt_path:-$teacher_ckpt_path_default}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Check if teacher checkpoint exists
if [ ! -f "$teacher_ckpt_path" ]; then
    echo "Error: Teacher checkpoint not found at $teacher_ckpt_path"
    exit 1
fi

mkdir -p $result_dir

# Run training
python /kaggle/working/KDFS/main.py \
    --phase train \
    --arch $arch \
    --device $device \
    --result_dir $result_dir \
    --teacher_ckpt_path "$teacher_ckpt_path" \
    --num_workers 4 \
    --pin_memory \
    --seed 3407 \
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
    --coef_rcloss 1.0 \
    --compress_rate 0.6 \
    "$@"

# Run finetuning
python /kaggle/working/KDFS/main.py \
    --phase finetune \
    --arch $arch \
    --device $device \
    --result_dir $result_dir \
    --finetune_student_ckpt_path $result_dir"/student_model/"$arch"_sparse_best.pt" \
    --num_workers 4 \
    --pin_memory \
    --seed 3407 \
    --finetune_num_epochs 6 \
    --finetune_lr 4e-6 \
    --finetune_warmup_steps 5 \
    --finetune_warmup_start_lr 4e-8 \
    --finetune_lr_decay_T_max 20 \
    --finetune_lr_decay_eta_min 4e-8 \
    --finetune_weight_decay 2e-5 \
    --finetune_train_batch_size 8 \
    --finetune_eval_batch_size 8 \
    --sparsed_student_ckpt_path $result_dir"/student_model/finetune_"$arch"_sparse_best.pt" \
    "$@"
