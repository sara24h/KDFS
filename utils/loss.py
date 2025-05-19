import torch
import torch.nn as nn
import torch.nn.functional as F


class KDLoss(nn.Module):
    def __init__(self):
        super(KDLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits_t, logits_s):
        return self.bce_loss(logits_s, torch.sigmoid(logits_t))  


class RCLoss(nn.Module):
    def __init__(self):
        super(RCLoss, self).__init__()

    @staticmethod
    def rc(x):
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

    def forward(self, x, y):
        return (self.rc(x) - self.rc(y)).pow(2).mean()




import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()

    def pearson_correlation(self, filters, mask):
        if filters.size(0) != mask.size(0):
            raise ValueError(f"Filters and mask must have same number of channels: {filters.size(0)} vs {mask.size(0)}")
        
        mask = mask.squeeze(-1).squeeze(-1).squeeze(-1)
        if mask.dim() != 1:
            raise ValueError(f"Mask must be squeezable to 1D, got shape: {mask.shape}")

        active_indices = torch.where(mask > 0)[0]
        if len(active_indices) == 0:
            return torch.zeros(filters.size(0), filters.size(0), device=filters.device, dtype=filters.dtype)
        elif len(active_indices) == 1:
            return torch.ones(1, 1, device=filters.device, dtype=filters.dtype)
        
        active_filters = filters[active_indices]
        flattened_filters = active_filters.view(active_filters.size(0), -1)

        if torch.isnan(flattened_filters).any() or torch.isinf(flattened_filters).any():
            return torch.zeros(filters.size(0), filters.size(0), device=filters.device, dtype=filters.dtype)

        # استفاده از autocast برای محاسبات ماتریس همبستگی
        with autocast():
            try:
                correlation_matrix = torch.corrcoef(flattened_filters)
            except RuntimeError:
                return torch.zeros(filters.size(0), filters.size(0), device=filters.device, dtype=filters.dtype)

        if correlation_matrix.dim() == 0:
            correlation_matrix = correlation_matrix.view(1, 1)

        if torch.isnan(correlation_matrix).any() or torch.isinf(correlation_matrix).any():
            return torch.zeros(filters.size(0), filters.size(0), device=filters.device, dtype=filters.dtype)

        full_correlation = torch.zeros(filters.size(0), filters.size(0), device=filters.device, dtype=filters.dtype)
        correlation_matrix = correlation_matrix.to(dtype=filters.dtype)
        full_correlation[active_indices[:, None], active_indices] = correlation_matrix

        return full_correlation

    def forward(self, weights, mask):
        # اطمینان از اینکه ورودی‌ها روی دستگاه مناسب (GPU) هستند
        if weights.device != mask.device:
            mask = mask.to(weights.device)

        # استفاده از autocast برای محاسبات پرهزینه
        with autocast():
            # محاسبه ماتریس همبستگی پیرسون
            correlation_matrix = self.pearson_correlation(weights, mask)
            mask = mask.squeeze(-1).squeeze(-1).squeeze(-1)
            
            # ساخت ماتریس ماسک
            mask_matrix = mask.unsqueeze(1) * mask.unsqueeze(0)
            
            # استخراج بخش بالا مثلثی ماتریس همبستگی (بدون قطر)
            upper_tri_pearson = torch.triu(correlation_matrix, diagonal=1)
            
            # اعمال ماتریس ماسک روی ماتریس بالا مثلثی
            masked_upper_tri = upper_tri_pearson * mask_matrix
            
            # محاسبه مجموع مربعات
            squared_sum = (masked_upper_tri ** 2).sum()
            
            # محاسبه تعداد عناصر فعال
            num_active = torch.triu(mask_matrix, diagonal=1).sum()
            
            # نرمال‌سازی loss
            if num_active > 0:
                normalized_loss = squared_sum / num_active
            else:
                normalized_loss = torch.tensor(0.0, device rood: weights.device, dtype=weights.dtype)
            
            # بررسی مقادیر نامعتبر
            if torch.isnan(normalized_loss) or torch.isinf(normalized_loss):
                normalized_loss = torch.tensor(0.0, device=weights.device, dtype=weights.dtype)
        
        return normalized_loss



class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss
