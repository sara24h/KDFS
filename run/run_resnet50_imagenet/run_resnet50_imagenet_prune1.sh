#!/bin/bash

# Default values
arch=ResNet_50
result_dir=/kaggle/working/results/run_resnet50_imagenet_prune1
device=cuda
teacher_ckpt_path_default=/kaggle/working/KDFS/teacher_dir/teacher_model_best.pth
resume_default=""

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
        *)
            shift
            ;;
    esac
done

# Use default values if not provided
teacher_ckpt_path=${teacher_ckpt_path:-$teacher_ckpt_path_default}
resume=${resume:-$resume_default}

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

mkdir -p $result_dir

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Run training
python /kaggle/working/KDFS/main.py \
    --phase train \
    --dataset_mode 140k \
    --dataset_dir /kaggle/input/140k-real-and-fake-faces \
    --realfake140k_train_csv /kaggle/input/140k-real-and-fake-faces/train.csv \
    --realfake140k_valid_csv /kaggle/input/140k-real-and-fake-faces/valid.csv \
    --realfake140k_test_csv /kaggle/input/140k-real-and-fake-faces/test.csv \
    --arch $arch \
    --device $device \
    --result_dir $result_dir \
    --teacher_ckpt_path "$teacher_ckpt_path" \
    --resume "$resume" \
    --num_workers 2 \
    --pin_memory \
    --seed 3407 \
    --num_epochs 10 \
    --lr 0.006 \
    --warmup_steps 10 \
    --warmup_start_lr 4e-5 \
    --lr_decay_T_max 250 \
    --lr_decay_eta_min 4e-5 \
    --weight_decay 5e-4 \
    --train_batch_size 128 \
    --eval_batch_size 32 \
    --test_batch_size 32 \
    --target_temperature 3 \
    --gumbel_start_temperature 1 \
    --gumbel_end_temperature 0.1 \
    --coef_kdloss 0.1 \
    --coef_rcloss 100 \
    --coef_maskloss 1000 \
    --compress_rate 0.68 \
    "$@"

# Run finetuning
python /kaggle/working/KDFS/main.py \
    --phase finetune \
    --dataset_mode 140k \
    --arch $arch \
    --device $device \
    --result_dir $result_dir \
    --teacher_ckpt_path "$teacher_ckpt_path" \
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
