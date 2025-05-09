#!/bin/bash

# Default values
arch=ResNet_50
result_dir=/kaggle/working/results/run_resnet50_imagenet_prune1
device=cuda
teacher_ckpt_path_default=/kaggle/working/KDFS/teacher_dir/teacher_model_best.pth
resume_default=""
dataset_mode_default="hardfake"
dataset_dir_default="/kaggle/input/hardfakevsrealfaces"

# Parse command-line arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --teacher_ckpt_path)
            teacher_ckpt_path="$2"
            shift 2
            ;;
        --resume)
            resume="$2"
            shift 2
            ;;
        --dataset_mode)
            dataset_mode="$2"
            shift 2
            ;;
        --dataset_dir)
            dataset_dir="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Use default values if not provided
teacher_ckpt_path=${teacher_ckpt_path:-$teacher_ckpt_path_default}
resume=${resume:-$resume_default}
dataset_mode=${dataset_mode:-$dataset_mode_default}
dataset_dir=${dataset_dir:-$dataset_dir_default}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TF_CUDNN_RESET_STATE=1

# Check if teacher checkpoint exists
if [ ! -f "$teacher_ckpt_path" ]; then
    echo "Error: Teacher checkpoint not found at $teacher_ckpt_path"
    exit 1
fi

# Check if resume checkpoint exists (if provided)
if [ -n "$resume" ] && [ ! -f "$resume" ]; then
    echo "Error: Resume checkpoint not found at $resume"
    exit 1
fi

# Check if dataset directory exists
if [ ! -d "$dataset_dir" ]; then
    echo "Error: Dataset directory not found at $dataset_dir"
    exit 1
fi

mkdir -p $result_dir

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Debug: Print arguments passed to training
echo "Training arguments: $@"
echo "Dataset mode: $dataset_mode"
echo "Dataset directory: $dataset_dir"

# Run training
python /kaggle/working/KDFS/main.py \
    --phase train \
    --dataset_mode "$dataset_mode" \
    --dataset_dir "$dataset_dir" \
    --arch "$arch" \
    --device "$device" \
    --result_dir "$result_dir" \
    --teacher_ckpt_path "$teacher_ckpt_path" \
    --resume "$resume" \
    --num_workers 2 \
    --pin_memory \
    --seed 3407 \
    --num_epochs 2 \
    --lr 4e-3 \
    --warmup_steps 10 \
    --warmup_start_lr 1e-05 \
    --lr_decay_T_max 250 \
    --lr_decay_eta_min 4e-5 \
    --weight_decay 5e-4 \
    --train_batch_size 64 \
    --eval_batch_size 32 \
    --test_batch_size 32 \
    --target_temperature 3 \
    --gumbel_start_temperature 1 \
    --gumbel_end_temperature 0.1 \
    --coef_kdloss 0.5 \
    --coef_rcloss 1 \
    --coef_maskloss 1 \
    --compress_rate 0.3 \
    "$@"
