import torch
import argparse
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal, ResNet_50_sparse_rvf10k
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
from model.student.MobileNet_sparse import MobileNetV2_sparse  # Assuming MobileNetV2_sparse is available
from model.pruned_model.MobileNet_pruned import MobileNetV2_pruned  # Assuming MobileNetV2_pruned is available
from thop import profile

# Base FLOPs and parameters for each dataset
Flops_baselines = {
    "ResNet_50": {
        "hardfakevsreal": 7700.0,
        "rvf10k": 5000.0,
        "140k": 5000.0,  # Added for consistency
    },
    "MobileNetV2": {
        "hardfakevsreal": 300.0,  # Approximate, adjust based on actual FLOPs
        "rvf10k": 200.0,          # Approximate, adjust based on actual FLOPs
        "140k": 200.0,            # Approximate, adjust based on actual FLOPs
    }
}
Params_baselines = {
    "ResNet_50": {
        "hardfakevsreal": 14.97,
        "rvf10k": 25.50,
        "140k": 25.50,  # Added for consistency
    },
    "MobileNetV2": {
        "hardfakevsreal": 2.23,  # Approximate, adjust based on actual params
        "rvf10k": 2.23,          # Approximate, adjust based on actual params
        "140k": 2.23,            # Approximate, adjust based on actual params
    }
}
image_sizes = {
    "hardfakevsreal": 300,
    "rvf10k": 256,
    "140k": 256,  # Added for consistency
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_mode",
        type=str,
        default="hardfake",
        choices=("hardfake", "rvf10k", "140k"),
        help="The type of dataset",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default=None,
        choices=("hardfakevsreal", "rvf10k", "140k", None),
        help="The dataset type for model naming",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="ResNet_50",
        choices=("ResNet_50", "MobileNetV2"),
        help="The architecture to prune",
    )
    parser.add_argument(
        "--sparsed_student_ckpt_path",
        type=str,
        default=None,
        help="The path where to load the sparsed student ckpt",
    )
    return parser.parse_args()

def calculate_baselines(arch, dataset_mode):
    dataset_type = "hardfakevsreal" if dataset_mode == "hardfake" else dataset_mode
    if arch == "ResNet_50":
        model = eval(arch + "_sparse_" + ("hardfakevsreal" if dataset_mode == "hardfake" else "rvf10k"))()
    else:  # MobileNetV2
        model = MobileNetV2_sparse(num_classes=1)
    input = torch.rand([1, 3, image_sizes[dataset_type], image_sizes[dataset_type]])
    flops, params = profile(model, inputs=(input,), verbose=False)
    return flops / (10**6), params / (10**6)

def get_flops_and_params(args):
    # Derive dataset_type from dataset_mode if not provided
    dataset_type = (
        "hardfakevsreal" if args.dataset_mode == "hardfake" else
        "rvf10k" if args.dataset_mode == "rvf10k" else
        "140k"
    )

    # Load sparse student model
    if args.arch == "ResNet_50":
        student = eval(args.arch + "_sparse_" + ("hardfakevsreal" if dataset_type == "hardfakevsreal" else "rvf10k"))()
    else:  # MobileNetV2
        student = MobileNetV2_sparse(num_classes=1)
    
    ckpt_student = torch.load(args.sparsed_student_ckpt_path, map_location="cpu", weights_only=True)
    student.load_state_dict(ckpt_student["student"])

    # Extract masks
    mask_weights = [m.mask_weight for m in student.mask_modules]
    masks = [
        torch.argmax(mask_weight, dim=1).squeeze(1).squeeze(1)
        for mask_weight in mask_weights
    ]

    # Load pruned model
    if args.arch == "ResNet_50":
        pruned_model = eval(args.arch + "_pruned_" + ("hardfakevsreal" if dataset_type == "hardfakevsreal" else "rvf10k"))(masks=masks)
    else:  # MobileNetV2
        pruned_model = MobileNetV2_pruned(masks=masks, num_classes=1)

    # Calculate FLOPs and Params
    input = torch.rand([1, 3, image_sizes[dataset_type], image_sizes[dataset_type]])
    Flops, Params = profile(pruned_model, inputs=(input,), verbose=False)

    # Use dataset-specific baseline values
    Flops_baseline = Flops_baselines[args.arch][dataset_type]
    Params_baseline = Params_baselines[args.arch][dataset_type]

    Flops_reduction = (
        (Flops_baseline - Flops / (10**6)) / Flops_baseline * 100.0
    )
    Params_reduction = (
        (Params_baseline - Params / (10**6)) / Params_baseline * 100.0
    )
    return (
        Flops_baseline,
        Flops / (10**6),
        Flops_reduction,
        Params_baseline,
        Params / (10**6),
        Params_reduction,
    )

def main():
    args = parse_args()

    # Calculate baselines for verification (uncomment to update Flops_baselines and Params_baselines)
    # flops_base, params_base = calculate_baselines(args.arch, args.dataset_mode)
    # print(f"Calculated Flops_baseline: {flops_base:.2f}M, Params_baseline: {params_base:.2f}M")

    (
        Flops_baseline,
        Flops,
        Flops_reduction,
        Params_baseline,
        Params,
        Params_reduction,
    ) = get_flops_and_params(args=args)
    print(
        f"[{args.arch}] Params_baseline: {Params_baseline:.2f}M, Params: {Params:.2f}M, Params reduction: {Params_reduction:.2f}%"
    )
    print(
        f"[{args.arch}] Flops_baseline: {Flops_baseline:.2f}M, Flops: {Flops:.2f}M, Flops reduction: {Flops_reduction:.2f}%"
    )

if __name__ == "__main__":
    main()
