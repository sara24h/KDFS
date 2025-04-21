import torch
import argparse
from thop import profile
from model.student.ResNet_sparse import ResNet_50_sparse_imagenet
from model.pruned_model.ResNet_pruned import ResNet_50_pruned_imagenet

# مقادیر پایه برای ResNet-50
Flops_baseline = 4134  # در میلیون
Params_baseline = 25.5  # در میلیون

class FlopsCalculator:
    def __init__(self, args):
        self.args = args
        self.arch = args.arch  # باید 'ResNet_50' باشد
        self.device = args.device
        self.sparsed_student_ckpt_path = args.sparsed_student_ckpt_path

    def build_model(self):
        print("==> Building sparse student model..")
        self.student = ResNet_50_sparse_imagenet()
        ckpt_student = torch.load(self.sparsed_student_ckpt_path, map_location="cpu")
        self.student.load_state_dict(ckpt_student["student"])
        print("Sparse student model loaded!")

    def calculate_flops_and_params(self):
        if self.device == "cuda":
            self.student = self.student.cuda()

        # استخراج ماسک‌ها
        mask_weights = [m.mask_weight for m in self.student.mask_modules]
        masks = [torch.argmax(mask_weight, dim=1).squeeze(1).squeeze(1) for mask_weight in mask_weights]

        # ساخت مدل pruned
        pruned_model = ResNet_50_pruned_imagenet(masks=masks)
        if self.device == "cuda":
            pruned_model = pruned_model.cuda()

        # محاسبه FLOPs و پارامترها
        input = torch.rand([1, 3, 224, 224], device=self.device)  # فرض اندازه تصویر 224x224
        Flops, Params = profile(pruned_model, inputs=(input,), verbose=False)
        Flops /= 1e6  # تبدیل به میلیون
        Params /= 1e6  # تبدیل به میلیون

        # محاسبه درصد کاهش
        Flops_reduction = (Flops_baseline - Flops) / Flops_baseline * 100.0
        Params_reduction = (Params_baseline - Params) / Params_baseline * 100.0

        print(f"Params_baseline: {Params_baseline:.2f}M, Params: {Params:.2f}M, Params reduction: {Params_reduction:.2f}%")
        print(f"Flops_baseline: {Flops_baseline:.2f}M, Flops: {Flops:.2f}M, Flops reduction: {Flops_reduction:.2f}%")

    def main(self):
        self.build_model()
        self.calculate_flops_and_params()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        type=str,
        default="ResNet_50",
        choices=["ResNet_50"],
        help="The architecture to calculate FLOPs and params for",
    )
    parser.add_argument(
        "--sparsed_student_ckpt_path",
        type=str,
        default=None,
        help="The path where to load the sparsed student checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    calculator = FlopsCalculator(args)
    calculator.main()
