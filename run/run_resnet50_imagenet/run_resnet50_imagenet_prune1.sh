
#!/bin/bash

# Model and dataset configurations
arch=ResNet_50
result_dir=result/run_resnet50_imagenet_prune1
dataset_dir=/kaggle/input/hardfakevsrealfaces
dataset_type=hardfakevsreal
csv_file=/kaggle/input/hardfakevsrealfaces/data.csv
teacher_ckpt_path=/kaggle/working/KDFS/teacher_dir/teacher_model.pth
device=cuda

# Run training
python /kaggle/working/KDFS/main.py \
    --phase train \
    --dataset_dir $dataset_dir \
    --dataset_type $dataset_type \
    --csv_file $csv_file \
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
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --target_temperature 3 \
    --gumbel_start_temperature 1 \
    --gumbel_end_temperature 0.1 \
    --coef_kdloss 0.05 \
    --coef_rcloss 1000 \
    --coef_maskloss 10000 \
    --compress_rate 0.4 \
&& \
# Run finetuning
python /kaggle/working/KDFS/main.py \
    --phase finetune \
    --dataset_dir $dataset_dir \
    --dataset_type $dataset_type \
    --csv_file $csv_file \
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
    --weight_decay 2e-5 \
    --finetune_train_batch_size 8 \
    --finetune_eval_batch_size 8 \
    --sparsed_student_ckpt_path $result_dir"/student_model/finetune_"$arch"_sparse_best.pt"
