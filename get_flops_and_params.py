import torch
from thop import profile
from model.student.ResNet_sparse import ResNet_50_sparse_imagenet
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_imagenet

# مقادیر پایه برای ResNet-50
Flops_baseline = 4134  # در میلیون
Params_baseline = 25.5  # در میلیون

def get_flops_and_params(args):
    """
    محاسبه FLOPs و پارامترهای مدل ResNet-50 sparse و pruned.
    
    Args:
        args: آرگومان‌ها شامل arch، device، dataset_type، و sparsed_student_ckpt_path
    
    Returns:
        tuple: (Flops_baseline, Flops, Flops_reduction, Params_baseline, Params, Params_reduction)
    """
    # بررسی معماری
    if args.arch != "ResNet_50":
        raise ValueError("This script only supports ResNet_50 architecture")

    # ساخت مدل sparse
    print("==> Building sparse student model..")
    student = ResNet_50_sparse_imagenet()
    ckpt_student = torch.load(args.sparsed_student_ckpt_path, map_location="cpu")
    student.load_state_dict(ckpt_student["student"])
    print("Sparse student model loaded!")

    # انتقال مدل به دستگاه مناسب
    device = args.device if hasattr(args, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        student = student.cuda()

    # استخراج ماسک‌ها
    mask_weights = [m.mask_weight for m in student.mask_modules]
    masks = [torch.argmax(mask_weight, dim=1).squeeze(1).squeeze(1) for mask_weight in mask_weights]

    # ساخت مدل pruned
    pruned_model = ResNet_50_pruned_imagenet(masks=masks)
    if device == "cuda":
        pruned_model = pruned_model.cuda()

    # تنظیم اندازه تصویر بر اساس dataset_type
    image_size = 224 if args.dataset_type == "hardfakevsrealfaces" else 32
    input = torch.rand([1, 3, image_size, image_size], device=device)
    
    # محاسبه FLOPs و پارامترها
    Flops, Params = profile(pruned_model, inputs=(input,), verbose=False)
    Flops /= 1e6  # تبدیل به میلیون
    Params /= 1e6  # تبدیل به میلیون

    # محاسبه درصد کاهش
    Flops_reduction = (Flops_baseline - Flops) / Flops_baseline * 100.0
    Params_reduction = (Params_baseline - Params) / Params_baseline * 100.0

    # چاپ نتایج
    print(f"Params_baseline: {Params_baseline:.2f}M, Params: {Params:.2f}M, Params reduction: {Params_reduction:.2f}%")
    print(f"Flops_baseline: {Flops_baseline:.2f}M, Flops: {Flops:.2f}M, Flops reduction: {Flops_reduction:.2f}%")

    return Flops_baseline, Flops, Flops_reduction, Params_baseline, Params, Params_reduction
