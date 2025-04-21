arch=ResNet_50
result_dir=result/run_resnet50_hardfakevsrealfaces
dataset_dir=/kaggle/input/hardfakevsrealfaces
dataset_type=hardfakevsrealfaces
csv_file=/kaggle/input/hardfakevsrealfaces/train.csv
teacher_dir=./teacher_dir

CUDA_VISIBLE_DEVICES=0 python main.py \
--phase train \
--dataset_dir $dataset_dir \
--dataset_type $dataset_type \
--csv_file $csv_file \
--num_workers 8 \
--pin_memory \
--device cuda \
--arch $arch \
--seed 3407 \
--result_dir $result_dir \
--teacher_dir $teacher_dir \
--num_epochs 50 \
--lr 1e-3 \
--warmup_steps 5 \
--warmup_start_lr 1e-5 \
--lr_decay_T_max 50 \
--lr_decay_eta_min 1e-5 \
--weight_decay 1e-4 \
--train_batch_size 64 \
--eval_batch_size 64
