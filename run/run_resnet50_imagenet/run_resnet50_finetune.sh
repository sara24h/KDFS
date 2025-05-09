#!/bin/bash

# Default values
arch_default="ResNet_50"
result_dir_default="result/run_resnet50_imagenet_prune1"
dataset_dir_default="dataset_imagenet"
dataset_type_default="imagenet"
teacher_ckpt_path_default="teacher_dir/resnet50-19c8e357.pth"
device_default="0,1,2,3"
master_port_default="6681"
num_workers_default=8
pin_memory_default="true"
seed_default=3407
num_epochs_default=250
lr_default=4e-3
warmup_steps_default=10
warmup_start_lr_default=4e-5
lr_decay_T_max_default=250
lr_decay_eta_min_default=4e-5
weight_decay_default=2e-5
train_batch_size_default=256
eval_batch_size_default=256
target_temperature_default=3
gumbel_start_temperature_default=1
gumbel_end_temperature_default=0.1
coef_kdloss_default=0.05
coef_rcloss_default=1000
coef_maskloss_default=10000
compress_rate_default=0.68
ddp_default="true"

# Parse command-line arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --arch)
            arch="$2"
            shift 2
            ;;
        --result_dir)
            result_dir="$2"
            shift 2
            ;;
        --dataset_dir)
            dataset_dir="$2"
            shift 2
            ;;
        --dataset_type)
            dataset_type="$2"
            shift 2
            ;;
        --teacher_ckpt_path)
            teacher_ckpt_path="$2"
            shift 2
            ;;
        --device)
            device="$2"
            shift 2
            ;;
        --master_port)
            master_port="$2"
            shift 2
            ;;
        --num_workers)
            num_workers="$2"
            shift 2
            ;;
        --pin_memory)
            pin_memory="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --num_epochs)
            num_epochs="$2"
            shift 2
            ;;
        --lr)
            lr="$2"
            shift 2
            ;;
        --warmup_steps)
            warmup_steps="$2"
            shift 2
            ;;
        --warmup_start_lr)
            warmup_start_lr="$2"
            shift 2
            ;;
        --lr_decay_T_max)
            lr_decay_T_max="$2"
            shift 2
            ;;
        --lr_decay_eta_min)
            lr_decay_eta_min="$2"
            shift 2
            ;;
        --weight_decay)
            weight_decay="$2"
            shift 2
            ;;
        --train_batch_size)
            train_batch_size="$2"
            shift 2
            ;;
        --eval_batch_size)
            eval_batch_size="$2"
            shift 2
            ;;
        --target_temperature)
            target_temperature="$2"
            shift 2
            ;;
        --gumbel_start_temperature)
            gumbel_start_temperature="$2"
            shift 2
            ;;
        --gumbel_end_temperature)
            gumbel_end_temperature="$2"
            shift 2
            ;;
        --coef_kdloss)
            coef_kdloss="$2"
            shift 2
            ;;
        --coef_rcloss)
            coef_rcloss="$2"
            shift 2
            ;;
        --coef_maskloss)
            coef_maskloss="$2"
            shift 2
            ;;
        --compress_rate)
            compress_rate="$2"
            shift 2
            ;;
        --ddp)
            ddp="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Use default values if not provided
arch=${arch:-$arch_default}
result_dir=${result_dir:-$result_dir_default}
dataset_dir=${dataset_dir:-$dataset_dir_default}
dataset_type=${dataset_type:-$dataset_type_default}
teacher_ckpt_path=${teacher_ckpt_path:-$teacher_ckpt_path_default}
device=${device:-$device_default}
master_port=${master_port:-$master_port_default}
num_workers=${num_workers:-$num_workers_default}
pin_memory=${pin_memory:-$pin_memory_default}
seed=${seed:-$seed_default}
num_epochs=${num_epochs:-$num_epochs_default}
lr=${lr:-$lr_default}
warmup_steps=${warmup_steps:-$warmup_steps_default}
warmup_start_lr=${warmup_start_lr:-$warmup_start_lr_default}
lr_decay_T_max=${lr_decay_T_max:-$lr_decay_T_max_default}
lr_decay_eta_min=${lr_decay_eta_min:-$lr_decay_eta_min_default}
weight_decay=${weight_decay:-$weight_decay_default}
train_batch_size=${train_batch_size:-$train_batch_size_default}
eval_batch_size=${eval_batch_size:-$eval_batch_size_default}
target_temperature=${target_temperature:-$target_temperature_default}
gumbel_start_temperature=${gumbel_start_temperature:-$gumbel_start_temperature_default}
gumbel_end_temperature=${gumbel_end_temperature:-$gumbel_end_temperature_default}
coef_kdloss=${coef_kdloss:-$coef_kdloss_default}
coef_rcloss=${coef_rcloss:-$coef_rcloss_default}
coef_maskloss=${coef_maskloss:-$coef_maskloss_default}
compress_rate=${compress_rate:-$compress_rate_default}
ddp=${ddp:-$ddp_default}

# Environment variables for CUDA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TF_CUDNN_RESET_STATE=1

# Check if teacher checkpoint exists
if [ ! -f "$teacher_ckpt_path" ]; then
    echo "Error: Teacher checkpoint not found at $teacher_ckpt_path"
    exit 1
fi

# Check if dataset directory exists
if [ ! -d "$dataset_dir" ]; then
    echo "Error: Dataset directory not found at $dataset_dir"
    exit 1
fi

# Create result directory
mkdir -p "$result_dir"

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Debug: Print training arguments
echo "Training arguments: $@"
echo "Architecture: $arch"
echo "Result directory: $result_dir"
echo "Dataset directory: $dataset_dir"
echo "Dataset type: $dataset_type"
echo "Teacher checkpoint: $teacher_ckpt_path"
echo "Device: $device"
echo "Master port: $master_port"
echo "Number of workers: $num_workers"
echo "Pin memory: $pin_memory"
echo "Seed: $seed"
echo "Number of epochs: $num_epochs"
echo "Learning rate: $lr"
echo "Warmup steps: $warmup_steps"
echo "Warmup start LR: $warmup_start_lr"
echo "LR decay T max: $lr_decay_T_max"
echo "LR decay eta min: $lr_decay_eta_min"
echo "Weight decay: $weight_decay"
echo "Train batch size: $train_batch_size"
echo "Eval batch size: $eval_batch_size"
echo "Target temperature: $target_temperature"
echo "Gumbel start temperature: $gumbel_start_temperature"
echo "Gumbel end temperature: $gumbel_end_temperature"
echo "Coefficient KD loss: $coef_kdloss"
echo "Coefficient RC loss: $coef_rcloss"
echo "Coefficient mask loss: $coef_maskloss"
echo "Compress rate: $compress_rate"
echo "DDP: $ddp"

# Run training with torchrun
CUDA_VISIBLE_DEVICES="$device" torchrun --nproc_per_node=4 --master_port "$master_port" main.py \
    --phase train \
    --dataset_dir "$dataset_dir" \
    --dataset_type "$dataset_type" \
    --num_workers "$num_workers" \
    $( [ "$pin_memory" = "true" ] && echo "--pin_memory" ) \
    --device cuda \
    --arch "$arch" \
    --seed "$seed" \
    --result_dir "$result_dir" \
    --teacher_ckpt_path "$teacher_ckpt_path" \
    --num_epochs "$num_epochs" \
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
    --coef_maskloss "$coef_maskloss" \
    --compress_rate "$compress_rate" \
    $( [ "$ddp" = "true" ] && echo "--ddp" ) \
    "$@"
