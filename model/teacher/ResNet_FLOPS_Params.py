import torch
from thop import profile
from ResNet import ResNet_50_hardfakevsreal
def calculate_flops_params(model, input_size=(1, 3, 300, 300), device='cuda'):
    """
    محاسبه FLOPs و تعداد پارامترهای یک مدل PyTorch.

    Args:
        model: مدل PyTorch (مثلاً ResNet_50_hardfakevsreal)
        input_size: ابعاد ورودی (batch_size, channels, height, width)
        device: دستگاه محاسباتی ('cuda' یا 'cpu')

    Returns:
        tuple: (flops, params) که flops تعداد عملیات ممیز شناور و params تعداد پارامترها است.
    """
    # تنظیم دستگاه
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = ResNet_50_hardfakevsreal().to(device)
    
    # ایجاد ورودی نمونه
    input_tensor = torch.randn(input_size).to(device)
    
    # محاسبه FLOPs و پارامترها
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    
    return flops, params
