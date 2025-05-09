#!/bin/bash

# Default values (aligned with training config from logs)
arch=${ARCH:-ResNet_50}
result_dir=${RESULT_DIR:-/kaggle/working/results/run_resnet50_imagenet_prune1}
teacher_ckpt_path=${TEACHER_CKPT_PATH:-/kaggle/working/KDFS/teacher_dir/teacher_model_best.pth}
device=${DEVICE:-cuda}
num_workers=${NUM_WORKERS:-4}
pin_memory=${PIN_MEMORY:-true}
seed=${SEED:-3407}
lr=${LR:-0.004}
warmup_steps=${WARMUP_STEPS:-10}
warmup_start_lr=${WARMUP_START_LR:-1e-05}
lr_decay_T_max=${LR_DECAY_T_MAX:-250}
lr_decay_eta_min=${LR_DECAY_ETA_MIN:-4e-05}
weight_decay=${WEIGHT_DECAY:-0.0005}
train_batch_size=${TRAIN_BATCH_SIZE:-16}
eval_batch_size=${EVAL_BATCH_SIZE:-16}
target_temperature=${TARGET_TEMPERATURE:-3}
gumbel_start_temperature=${GUMBEL_START_TEMPERATURE:-1}
gumbel_end_temperature=${GUMBEL_END_TEMPERATURE:-0.1}
coef_kdloss=${COEF_KDLOSS:-0.5}
coef_rcloss=${COEF_RCLOSS:-1.0}
compress_rate=${COMPRESS_RATE:-0.3}
finetune_num_epochs=${FINETUNE_NUM_EPOCHS:-6}
finetune_lr=${FINETUNE_LR:-4e-06}
finetune_warmup_steps=${FINETUNE_WARMUP_STEPS:-5}
finetune_warmup_start_lr=${FINETUNE_WARMUP_START_LR:-4e-08}
finetune_lr_decay_T_max=${FINETUNE_LR_DECAY_T_MAX:-20}
finetune_lr_decay_eta_min=${FINETUNE_LR_DECAY_ETA_MIN:-4e-08}
finetune_weight_decay=${FINETUNE_WEIGHT_DECAY:-2e-05}
finetune_train_batch_size=${FINETUNE_TRAIN_BATCH_SIZE:-8}
finetune_eval_batch_size=${FINETUNE_EVAL_BATCH_SIZE:-8}

# Environment variables for CUDA and memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # Reduce CUDA factory registration warnings

# Check if teacher checkpoint exists
if [ ! -f "$teacher_ckpt_path" ]; then
    echo "Error: Teacher checkpoint not found at $teacher_ckpt_path"
    exit 1
fi

# Create result directory
mkdir -p "$result_dir"

# Run training
python /kaggle/working/KDFS/main.py \
    --phase train \
    --arch "$arch" \
    --device "$device" \
    --result_dir "$result_dir" \
    --teacher_ckpt_path "$teacher_ckpt_path" \
    --num_workers "$num_workers" \
    $( [ "$pin_memory" = "true" ] && echo "--pin_memory" ) \
    --seed "$seed" \
    --lr "$lr" \
    --warmup_steps "$warmup_steps" \
    --warmup_start_lr "$warmup_start_lr" \
    --lr_decay_T_max "$lr_decay_T_max" \
    --lr_decay_eta_min "$lr_decay_eta_min" \
    --weight_decay "$weight_decay" \
    --train_batch_size "$train_batch_size" \
    --eval_batch_size "$eval_batch_size" \
    --target_temperature "$target_temperature" \
    --gumbel_start_temperature "$gumbel_start_temperature" \
    --gumbel_end_temperature "$gumbel_end_temperature" \
    --coef_kdloss "$coef_kdloss" \
    --coef_rcloss "$coef_rcloss" \
    --compress_rate "$compress_rate" \
    "$@"

# Check if student checkpoint exists before finetuning
student_ckpt_path="$result_dir/student_model/${arch}_sparse_best.pt"
if [ ! -f "$student_ckpt_path" ]; then
    echo "Error: Student checkpoint not found at $student_ckpt_path"
    exit 1
fi

# Run finetuning
python /kaggle/working/KDFS/main.py \
    --phase finetune \
    --arch "$arch" \
    --device "$device" \
    --result_dir "$result_dir" \
    --teacher_ckpt_path "$teacher_ckpt_path" \
    --finetune_student_ckpt_path "$student_ckpt_path" \
    --num_workers "$num_workers" \
    $( [ "$pin_memory" = "true" ] && echo "--pin_memory" ) \
    --seed "$seed" \
    --finetune_num_epochs "$finetune_num_epochs" \
    --finetune_lr "$finetune_lr" \
    --finetune_warmup_steps "$finetune_warmup_steps" \
    --finetune_warmup_start_lr "$finetune_warmup_start_lr" \
    --finetune_lr_decay_T_max "$finetune_lr_decay_T_max" \
    --finetune_lr_decay_eta_min "$finetune_lr_decay_eta_min" \
    --finetune_weight_decay "$finetune_weight_decay" \
    --finetune_train_batch_size "$finetune_train_batch_size" \
    --finetune_eval_batch_size "$finetune_eval_batch_size" \
    --sparsed_student_ckpt_path "$result_dir/student_model/finetune_${arch}_sparse_best.pt" \
    "$@"
