arch=ResNet_50
result_dir=results/run_resnet50_hardfakevsrealfaces
dataset_dir=/kaggle/input/hardfakevsrealfaces
dataset_type=hardfakevsrealfaces
csv_file=/kaggle/input/hardfakevsrealfaces/data.csv
teacher_dir=/kaggle/working/KDFS/teacher_dir

CUDA_VISIBLE_DEVICES=0 python main.py \
--phase train \
--dataset_dir $dataset_dir \
--dataset_type $dataset_type \
--num_workers 4 \
--csv_file /kaggle/input/hardfakevsrealfaces/data.csv \
--teacher_dir /kaggle/working/KDFS/teacher_dir \
--teacher_ckpt_path /kaggle/working/KDFS/teacher_dir/teacher_model.pth \
--pin_memory \
--device cuda \
--arch $arch \
--seed 42 \
--result_dir $result_dir \
--num_epochs 50 \
--lr 0.001 \
--warmup_steps 5 \
--warmup_start_lr 0.0001 \
--lr_decay_T_max 50 \
--lr_decay_eta_min 0.00001 \
--weight_decay 5e-4 \
--train_batch_size 32 \
--eval_batch_size 32
