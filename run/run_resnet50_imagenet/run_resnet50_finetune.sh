#!/bin/bash

# Default values for finetune
arch=ResNet_50
dataset_mode=140k
dataset_dir=/kaggle/input/140k-real-and-fake-faces
realfake140k_train_csv=/kaggle/input/140k-real-and-fake-faces/train.csv
realfake140k_valid_csv=/kaggle/input/140k-real-and-fake-faces/valid.csv
realfake140k_test_csv=/kaggle/input/140k-real-and-fake-faces/test.csv
result_dir=/kaggle/working/results/run_resnet50_imagenet_prune1
device=cuda
teacher_ckpt_path_default=/kaggle/working/KDFS/teacher_dir/teacher_model_best.pth
finetune_student_ckpt_path_default=$result_dir"/student_model/"$arch"_sparse_best.pt"
sparsed_student_ckpt_path_default=$result_dir"/student_model/finetune_"$arch"_sparse_best.pt"
num_workers=4
seed=3407
finetune_num_epochs=3
finetune_lr=4e-6
finetune_warmup_steps=5
finetune_warmup_start_lr=4e-8
finetune_lr_decay_T_max=20
finetune_lr_decay_eta_min=4e-8
finetune_weight_decay=2e-5
finetune_train_batch_size=8
finetune_eval_batch_size=8
pin_memory=true

# Parse command-line arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --arch) arch="$2"; shift 2 ;;
        --dataset_mode) dataset_mode="$2"; shift 2 ;;
        --dataset_dir) dataset_dir="$2"; shift 2 ;;
        --realfake140k_train_csv) realfake140k_train_csv="$2"; shift 2 ;;
        --realfake140k_valid_csv) realfake140k_valid_csv="$2"; shift 2 ;;
        --realfake140k_test_csv) realfake140k_test_csv="$2"; shift 2 ;;
        --result_dir) result_dir="$2"; shift 2 ;;
        --device) device="$2"; shift 2 ;;
        --teacher_ckpt_path) teacher_ckpt_path="$2"; shift 2 ;;
        --finetune_student_ckpt_path) finetune_student_ckpt_path="$2"; shift 2 ;;
        --sparsed_student_ckpt_path) sparsed_student_ckpt_path="$2"; shift 2 ;;
        --num_workers) num_workers="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
        --finetune_num_epochs) finetune_num_epochs="$2"; shift 2 ;;
        --finetune_lr) finetune_lr="$2"; shift 2 ;;
        --finetune_warmup_steps) finetune_warmup_steps="$2"; shift 2 ;;
        --finetune_warmup_start_lr) finetune_warmup_start_lr="$2"; shift 2 ;;
        --finetune_lr_decay_T_max) finetune_lr_decay_T_max="$2"; shift 2 ;;
        --finetune_lr_decay_eta_min) finetune_lr_decay_eta_min="$2"; shift 2 ;;
        --finetune_weight_decay) finetune_weight_decay="$2"; shift 2 ;;
        --finetune_train_batch_size) finetune_train_batch_size="$2"; shift 2 ;;
        --finetune_eval_batch_size) finetune_eval_batch_size="$2"; shift 2 ;;
        --pin_memory) pin_memory="$2"; shift 2 ;;
        *) shift ;;  # Ignore unknown arguments
    esac
done

# Use default values if not provided
teacher_ckpt_path=${teacher_ckpt_path:-$teacher_ckpt_path_default}
finetune_student_ckpt_path=${finetune_student_ckpt_path:-$finetune_student_ckpt_path_default}
sparsed_student_ckpt_path=${sparsed_student_ckpt_path:-$sparsed_student_ckpt_path_default}

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
if [ ! -f "$realfake140k_train_csv" ] || [ ! -f "$realfake140k_valid_csv" ] || [ ! -f "$realfake140k_test_csv" ]; then
    echo "Error: One or more dataset CSV files not found"
    exit 1
fi

# Create result directory
mkdir -p $result_dir

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Debug: Print finetuning arguments
echo "Finetuning arguments:"
echo "arch=$arch"
echo "dataset_mode=$dataset_mode"
echo "dataset_dir=$dataset_dir"
echo "realfake140k_train_csv=$realfake140k_train_csv"
echo "realfake140k_valid_csv=$realfake140k_valid_csv"
echo "realfake140k_test_csv=$realfake140k_test_csv"
echo "result_dir=$result_dir"
echo "device=$device"
echo "teacher_ckpt_path=$teacher_ckpt_path"
echo "finetune_student_ckpt_path=$finetune_student_ckpt_path"
echo "sparsed_student_ckpt_path=$sparsed_student_ckpt_path"
echo "num_workers=$num_workers"
echo "seed=$seed"
echo "finetune_num_epochs=$finetune_num_epochs"
echo "finetune_lr=$finetune_lr"
echo "finetune_warmup_steps=$finetune_warmup_steps"
echo "finetune_warmup_start_lr=$finetune_warmup_start_lr"
echo "finetune_lr_decay_T_max=$finetune_lr_decay_T_max"
echo "finetune_lr_decay_eta_min=$finetune_lr_decay_eta_min"
echo "finetune_weight_decay=$finetune_weight_decay"
echo "finetune_train_batch_size=$finetune_train_batch_size"
echo "finetune_eval_batch_size=$finetune_eval_batch_size"
echo "pin_memory=$pin_memory"

# Run finetuning
python /kaggle/working/KDFS/main.py \
    --phase finetune \
    --dataset_mode "$dataset_mode" \
    --dataset_dir "$dataset_dir" \
    --realfake140k_train_csv "$realfake140k_train_csv" \
    --realfake140k_valid_csv "$realfake140k_valid_csv" \
    --realfake140k_test_csv "$realfake140k_test_csv" \
    --arch "$arch" \
    --device "$device" \
    --result_dir "$result_dir" \
    --teacher_ckpt_path "$teacher_ckpt_path" \
    --finetune_student_ckpt_path "$finetune_student_ckpt_path" \
    --sparsed_student_ckpt_path "$sparsed_student_ckpt_path" \
    --num_workers "$num_workers" \
    --pin_memory \
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
    "$@"
