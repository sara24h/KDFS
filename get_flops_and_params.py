import torch
import argparse
from thop import profile

# فرض می‌کنیم این ماژول‌ها به درستی تعریف شده‌اند
from model.student.ResNet_sparse import ResNet_50_sparse_imagenet
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_imagenet

Flops_baselines = {
    "ResNet_50": 4134,  # FLOPs پایه برای ResNet_50 (میلیون)
}
Params_baselines = {
    "ResNet_50": 25.50,  # پارامترهای پایه برای ResNet_50 (میلیون)
}
image_sizes = {"imagenet": 224}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="hardfakevsrealfaces",
        choices=("imagenet",),
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

def get_flops_and_params(args):
    # بارگذاری مدل sparse
    student = ResNet_50_sparse_imagenet()
    ckpt_student = torch.load(args.sparsed_student_ckpt_path, map_location="cpu")
    student.load_state_dict(ckpt_student["student"])

    # استخراج ماسک‌ها
    mask_weights = [m.mask_weight for m in student.mask_modules]
    masks = [
        torch.argmax(mask_weight, dim=1).squeeze(1).squeeze(1)
        for mask_weight in mask_weights
    ]

    # بارگذاری مدل pruned
    pruned_model = ResNet_50_pruned_imagenet(masks=masks)
    input = torch.rand([1, 3, image_sizes[args.dataset_type], image_sizes[args.dataset_type]])

    # محاسبه FLOPs و پارامترها
    Flops, Params = profile(pruned_model, inputs=(input,), verbose=False)
    Flops_reduction = (
        (Flops_baselines[args.arch] - Flops / (10**6))
        / Flops_baselines[args.arch]
        * 100.0
    )
    Params_reduction = (
        (Params_baselines[args.arch] - Params / (10**6))
        / Params_baselines[args.arch]
        * 100.0
    )
    return (
        Flops_baselines[args.arch],
        Flops / (10**6),
        Flops_reduction,
        Params_baselines[args.arch],
        Params / (10**6),
        Params_reduction,
    )

def main():
    args = parse_args()
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
