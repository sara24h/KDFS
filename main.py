import argparse
import os
from train import Train
from finetune import Finetune  # Assuming your fine-tuning script is named fine_tune.py

def parse_args():
    desc = "Pytorch implementation of KDFS"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--phase",
        type=str,
        default="train",
        choices=("train", "finetune"),
        help="train or finetune",
    )

    # Common
    parser.add_argument(
        "--dataset_dir", type=str, default="/kaggle/input/hardfakevsrealfaces", help="The dataset path"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="hardfakevsrealfaces",
        choices=("cifar10", "hardfakevsrealfaces"),
        help="The type of dataset",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,  # Reduced for memory efficiency
        help="The num_workers of dataloader",
    )
    parser.add_argument(
        "--pin_memory",
        type=bool,
        default=False,  # Disabled for memory efficiency
        help="The pin_memory of dataloader",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="ResNet_50",
        choices=("ResNet_50", "resnet_56"),
        help="The architecture to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu"),
        help="Device to use",
    )
    parser.add_argument("--seed", type=int, default=42, help="Init seed")
    parser.add_argument(
        "--result_dir",
        type=str,
        default="./results/",
        help="The directory where the results will be stored",
    )

    # Train
    parser.add_argument(
        "--teacher_ckpt_path",
        type=str,
        default="teacher_resnet50_finetuned.pth",
        help="The path where to load the teacher ckpt",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="The num of epochs to train"
    )
    parser.add_argument(
        "--lr", default=1e-4, type=float, help="The initial learning rate of model"
    )
    parser.add_argument(
        "--warmup_steps",
        default=5,
        type=int,
        help="The steps of warmup",
    )
    parser.add_argument(
        "--warmup_start_lr",
        default=1e-6,
        type=float,
        help="The starting learning rate for warmup",
    )
    parser.add_argument(
        "--lr_decay_T_max",
        default=10,
        type=int,
        help="T_max of CosineAnnealingLR",
    )
    parser.add_argument(
        "--lr_decay_eta_min",
        default=1e-6,
        type=float,
        help="eta_min of CosineAnnealingLR",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="Batch size for validation"
    )
    parser.add_argument(
        "--accumulation_steps", type=int, default=2, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--target_temperature",
        type=float,
        default=4.0,
        help="Temperature of soft targets",
    )
    parser.add_argument(
        "--gumbel_start_temperature",
        type=float,
        default=1.0,
        help="Gumbel-softmax temperature at the start of training",
    )
    parser.add_argument(
        "--gumbel_end_temperature",
        type=float,
        default=0.1,
        help="Gumbel-softmax temperature at the end of training",
    )
    parser.add_argument(
        "--coef_kdloss", type=float, default=1.0, help="Coefficient of kd loss"
    )
    parser.add_argument(
        "--coef_rcloss",
        type=float,
        default=0.0,
        help="Coefficient of reconstruction loss",
    )
    parser.add_argument(
        "--coef_maskloss", type=float, default=0.0, help="Coefficient of mask loss"
    )
    parser.add_argument(
        "--compress_rate",
        type=float,
        default=0.5,
        help="Compress rate of the student model",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Load the model from the specified checkpoint",
    )

    # Finetune
    parser.add_argument(
        "--finetune_num_epochs",
        type=int,
        default=10,
        help="The num of epochs to train in finetune",
    )
    parser.add_argument(
        "--finetune_lr",
        default=1e-4,
        type=float,
        help="The initial learning rate of model in finetune",
    )
    parser.add_argument(
        "--finetune_warmup_steps",
        default=5,
        type=int,
        help="The steps of warmup in finetune",
    )
    parser.add_argument(
        "--finetune_warmup_start_lr",
        default=1e-6,
        type=float,
        help="The starting learning rate for warmup in finetune",
    )
    parser.add_argument(
        "--finetune_lr_decay_T_max",
        default=10,
        type=int,
        help="T_max of CosineAnnealingLR in finetune",
    )
    parser.add_argument(
        "--finetune_lr_decay_eta_min",
        default=1e-6,
        type=float,
        help="eta_min of CosineAnnealingLR in finetune",
    )
    parser.add_argument(
        "--finetune_weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay in finetune",
    )
    parser.add_argument(
        "--finetune_train_batch_size",
        type=int,
        default=16,
        help="Batch size for training in finetune",
    )
    parser.add_argument(
        "--finetune_eval_batch_size",
        type=int,
        default=16,
        help="Batch size for validation in finetune",
    )
    parser.add_argument(
        "--finetune_resume",
        type=str,
        default=None,
        help="Load the model from the specified checkpoint in finetune",
    )

    return parser.parse_args()

def main():
    args = parse_args()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["OMP_NUM_THREADS"] = "4"

    if args.phase == "train":
        train = Train(args=args)
        train.main()
    elif args.phase == "finetune":
        finetune = Finetune(args=args)
        finetune.main()
    else:
        raise ValueError(f"Unsupported phase: {args.phase}")

if __name__ == "__main__":
    main()
