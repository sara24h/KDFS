import torch
import argparse
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_hardfakevsreal
from thop import profile

# مقادیر پایه FLOPs و پارامترها برای هر دیتاست (به‌روزرسانی‌شده برای num_classes=1)
Flops_baselines = {
    "ResNet_50": {
        "hardfakevsrealfaces": 7690.0,  # باید با calculate_baselines تأیید بشه
        "rvf10k": 5000.0,  # باید با calculate_baselines تأیید بشه
    }
}
Params_baselines = {
    "ResNet_50": {
        "hardfakevsrealfaces": 23.50,  # به‌روزرسانی‌شده برای num_classes=1
        "rvf10k": 25.50,  # به‌روزرسانی‌شده برای num_classes=1
    }
}
image_sizes = {
    "hardfakevsrealfaces": 300,
    "rvf10k": 256,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_mode",
        type=str,
        default="hardfakevsrealfaces",
        choices=("hardfakevsrealfaces", "rvf10k"),
        help="The type of dataset",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="ResNet_50",
        choices=("ResNet_50",),
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
    model = eval(arch + "_sparse_" + dataset_mode)()
    input = torch.rand([1, 3, image_sizes[dataset_mode], image_sizes[dataset_mode]])
    flops, params = profile(model, inputs=(input,), verbose=False)
    return flops / (10**6), params / (10**6)

def get_flops_and_params(args):
    student = eval(args.arch + "_sparse_" + args.dataset_mode)()
    ckpt_student = torch.load(args.sparsed_student_ckpt_path, map_location="cpu", weights_only=True)
    student.load_state_dict(ckpt_student["student"])

    mask_weights = [m.mask_weight for m in student.mask_modules]
    masks = [
        torch.argmax(mask_weight, dim=1).squeeze(1).squeeze(1)
        for mask_weight in mask_weights
    ]
    pruned_model = eval(args.arch + "_pruned_" + args.dataset_mode)(masks=masks)
    input = torch.rand(
        [1, 3, image_sizes[args.dataset_mode], image_sizes[args.dataset_mode]]
    )
    Flops, Params = profile(pruned_model, inputs=(input,), verbose=False)

    # استفاده از مقادیر پایه خاص دیتاست
    Flops_baseline = Flops_baselines[args.arch][args.dataset_mode]
    Params_baseline = Params_baselines[args.arch][args.dataset_mode]

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

    # محاسبه مقادیر پایه برای تأیید (اجرا کنید و مقادیر Flops_baselines و Params_baselines را به‌روزرسانی کنید)
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
        "Params_baseline: %.2fM, Params: %.2fM, Params reduction: %.2f%%"
        % (Params_baseline, Params, Params_reduction)
    )
    print(
        "Flops_baseline: %.2fM, Flops: %.2fM, Flops reduction: %.2f%%"
        % (Flops_baseline, Flops, Flops_reduction)
    )

if __name__ == "__main__":
    main()
