#!/bin/bash

# Default values for finetune
arch_default="ResNet_50"
dataset_mode_default="hardfake"
dataset_dir_default="/kaggle/input/hardfakevsrealfaces"
hardfake_csv_file_default="/kaggle/input/hardfakevsrealfaces/data.csv"
realfake140k_train_csv_default="/kaggle/input/140k-real-and-fake-faces/train.csv"
realfake140k_valid_csv_default="/kaggle/input/140k-real-and-fake-faces/valid.csv"
realfake140k_test_csv_default="/kaggle/input/140k-real-and-fake-faces/test.csv"
result_dir_default="/kaggle/working/results/run_resnet50_imagenet_prune1"
device_default="cuda"
teacher_ckpt_path_default="/kaggle/working/KDFS/teacher_dir/teacher_model_best.pth"
finetune_student_ckpt_path_default="${result_dir_default}/student_model/${arch_default}_sparse_best.pt"
sparsed_student_ckpt_path_default="${result_dir_default}/student_model/finetune_${arch_default}_sparse_best.pt"
num_workers_default=2
seed_default=3407
finetune_num_epochs_default=1
finetune_lr_default=4e-6
finetune_warmup_steps_default=5
finetune_warmup_start_lr_default=4e-8
finetune_lr_decay_T_max_default=20
finetune_lr_decay_eta_min_default=4e-8
finetune_weight_decay_default=2e-5
finetune_train_batch_size_default=4
finetune_eval_batch_size_default=4
pin_memory_default="true"
test_batch_size_default=16
master_port_default="6681"

# Parse command-line arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --arch)
            arch="$2"
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
        --hardfake_csv_file)
            hardfake_csv_file="$2"
            shift 2
            ;;
        --realfake140k_train_csv)
            realfake140k_train_csv="$2"
            shift 2
            ;;
        --realfake140k_valid_csv)
            realfake140k_valid_csv="$2"
            shift 2
            ;;
        --realfake140k_test_csv)
            realfake140k_test_csv="$2"
            shift 2
            ;;
        --result_dir)
            result_dir="$2"
            shift 2
            ;;
        --device)
            device="$2"
            shift 2
            ;;
        --teacher_ckpt_path)
            teacher_ckpt_path="$2"
            shift 2
            ;;
        --finetune_student_ckpt_path)
            finetune_student_ckpt_path="$2"
            shift 2
            ;;
        --sparsed_student_ckpt_path)
            sparsed_student_ckpt_path="$2"
            shift 2
            ;;
        --num_workers)
            num_workers="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --finetune_num_epochs)
            finetune_num_epochs="$2"
            shift 2
            ;;
        --finetune_lr)
            finetune_lr="$2"
            shift 2
            ;;
        --finetune_warmup_steps)
            finetune_warmup_steps="$2"
            shift 2
            ;;
        --finetune_warmup_start_lr)
            finetune_warmup_start_lr="$2"
            shift 2
            ;;
        --finetune_lr_decay_T_max)
            finetune_lr_decay_T_max="$2"
            shift 2
            ;;
        --finetune_lr_decay_eta_min)
            finetune_lr_decay_eta_min="$2"
            shift 2
            ;;
        --finetune_weight_decay)
            finetune_weight_decay="$2"
            shift 2
            ;;
        --finetune_train_batch_size)
            finetune_train_batch_size="$2"
            shift 2
            ;;
        --finetune_eval_batch_size)
            finetune_eval_batch_size="$2"
            shift 2
            ;;
        --pin_memory)
            pin_memory="$2"
            shift 2
            ;;
        --test_batch_size)
            test_batch_size="$2"
            shift 2
            ;;
        --master_port)
            master_port="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Use default values if not provided
arch=${arch:-$arch_default}
dataset_mode=${dataset_mode:-$dataset_mode_default}
dataset_dir=${dataset_dir:-$dataset_dir_default}
hardfake_csv_file=${hardfake_csv_file:-$hardfake_csv_file_default}
realfake140k_train_csv=${realfake140k_train_csv:-$realfake140k_train_csv_default}
realfake140k_valid_csv=${realfake140k_valid_csv:-$realfake140k_valid_csv_default}
realfake140k_test_csv=${realfake140k_test_csv:-$realfake140k_test_csv_default}
result_dir=${result_dir:-$result_dir_default}
device=${device:-$device_default}
teacher_ckpt_path=${teacher_ckpt_path:-$teacher_ckpt_path_default}
finetune_student_ckpt_path=${finetune_student_ckpt_path:-$finetune_student_ckpt_path_default}
sparsed_student_ckpt_path=${sparsed_student_ckpt_path:-$sparsed_student_ckpt_path_default}
num_workers=${num_workers:-$num_workers_default}
seed=${seed:-$seed_default}
finetune_num_epochs=${finetune_num_epochs:-$finetune_num_epochs_default}
finetune_lr=${finetune_lr:-$finetune_lr_default}
finetune_warmup_steps=${finetune_warmup_steps:-$finetune_warmup_steps_default}
finetune_warmup_start_lr=${finetune_warmup_start_lr:-$finetune_warmup_start_lr_default}
finetune_lr_decay_T_max=${finetune_lr_decay_T_max:-$finetune_lr_decay_T_max_default}
finetune_lr_decay_eta_min=${finetune_lr_decay_eta_min:-$finetune_lr_decay_eta_min_default}
finetune_weight_decay=${finetune_weight_decay:-$finetune_weight_decay_default}
finetune_train_batch_size=${finetune_train_batch_size:-$finetune_train_batch_size_default}
finetune_eval_batch_size=${finetune_eval_batch_size:-$finetune_eval_batch_size_default}
pin_memory=${pin_memory:-$pin_memory_default}
test_batch_size=${test_batch_size:-$test_batch_size_default}
master_port=${master_port:-$master_port_default}

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

# Check if finetune student checkpoint exists
if [ ! -f "$finetune_student_ckpt_path" ]; then
    echo "Error: Finetune student checkpoint not found at $finetune_student_ckpt_path"
    exit 1
fi

# Check if dataset files exist
if [ "$dataset_mode" = "hardfake" ] && [ ! -f "$hardfake_csv_file" ]; then
    echo "Error: Hardfake CSV file not found at $hardfake_csv_file"
    exit 1
elif [ "$dataset_mode" = "140k" ] && { [ ! -f "$realfake140k_train_csv" ] || [ ! -f "$realfake140k_valid_csv" ] || [ ! -f "$realfake140k_test_csv" ]; }; then
    echo "Error: One or more 140k dataset CSV files not found"
    exit 1
fi

# Create result directory
mkdir -p "$result_dir"

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Debug: Print finetuning arguments
echo "Finetuning arguments: $@"
echo "Architecture: $arch"
echo "Dataset mode: $dataset_mode"
echo "Dataset directory: $dataset_dir"
echo "Hardfake CSV: $hardfake_csv_file"
echo "Train CSV (140k): $realfake140k_train_csv"
echo "Valid CSV (140k): $realfake140k_valid_csv"
echo "Test CSV (140k): $realfake140k_test_csv"
echo "Result directory: $result_dir"
echo "Device: $device"
echo "Teacher checkpoint: $teacher_ckpt_path"
echo "Finetune student checkpoint: $finetune_student_ckpt_path"
echo "Sparsed student checkpoint: $sparsed_student_ckpt_path"
echo "Number of workers: $num_workers"
echo "Seed: $seed"
echo "Finetune number of epochs: $finetune_num_epochs"
echo "Finetune learning rate: $finetune_lr"
echo "Finetune warmup steps: $finetune_warmup_steps"
echo "Finetune warmup start LR: $finetune_warmup_start_lr"
echo "Finetune LR decay T max: $finetune_lr_decay_T_max"
echo "Finetune LR decay eta min: $finetune_lr_decay_eta_min"
echo "Finetune weight decay: $finetune_weight_decay"
echo "Finetune train batch size: $finetune_train_batch_size"
echo "Finetune eval batch size: $finetune_eval_batch_size"
echo "Test batch size: $test_batch_size"
echo "Pin memory: $pin_memory"
echo "Master port: $master_port"

# Detect number of available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l 2>/dev/null || echo 0)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No GPUs detected. Falling back to CPU."
    DEVICE="cpu"
    NPROC_PER_NODE=1
    RUN_COMMAND="python"
else
    echo "Detected $NUM_GPUS GPUs."
    DEVICE="cuda"
    NPROC_PER_NODE=$NUM_GPUS
    RUN_COMMAND="torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$master_port"
fi

# Run finetuning
$RUN_COMMAND /kaggle/working/KDFS/main.py \
    --phase finetune \
    --dataset_mode "$dataset_mode" \
    --dataset_dir "$dataset_dir" \
    --hardfake_csv_file "$hardfake_csv_file" \
    --realfake140k_train_csv "$realfake140k_train_csv" \
    --realfake140k_valid_csv "$realfake140k_valid_csv" \
    --realfake140k_test_csv "$realfake140k_test_csv" \
    --arch "$arch" \
    --device "$DEVICE" \
    --result_dir "$result_dir" \
    --teacher_ckpt_path "$teacher_ckpt_path" \
    --finetune_student_ckpt_path "$finetune_student_ckpt_path" \
    --sparsed_student_ckpt_path "$sparsed_student_ckpt_path" \
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
    --test_batch_size "$test_batch_size" \
    "$@"
