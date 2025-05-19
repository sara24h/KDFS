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
        

class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()

    def pearson_correlation(self, filters, mask):
        # بررسی سازگاری ابعاد
        if filters.size(0) != mask.size(0):
            raise ValueError(f"Filters and mask must have same number of channels: {filters.size(0)} vs {mask.size(0)}")
        
        # فشرده‌سازی ماسک به 1D
        mask = mask.squeeze(-1).squeeze(-1).squeeze(-1)
        if mask.dim() != 1:
            raise ValueError(f"Mask must be squeezable to 1D, got shape: {mask.shape}")

        # یافتن ایندکس‌های فعال
        active_indices = torch.where(mask > 0)[0]
        if len(active_indices) == 0:
            return torch.zeros(filters.size(0), filters.size(0), device=filters.device, dtype=filters.dtype)
        elif len(active_indices) == 1:
            return torch.ones(1, 1, device=filters.device, dtype=filters.dtype)

        # انتخاب فیلترهای فعال
        active_filters = filters[active_indices]
        flattened_filters = active_filters.view(active_filters.size(0), -1)

        # بررسی مقادیر نامعتبر (NaN یا Inf)
        if torch.isnan(flattened_filters).any() or torch.isinf(flattened_filters).any():
            return torch.zeros(filters.size(0), filters.size(0), device=filters.device, dtype=filters.dtype)

        # محاسبه میانگین و مرکز کردن داده‌ها
        mean = flattened_filters.mean(dim=1, keepdim=True)
        centered = flattened_filters - mean

        # محاسبه کوواریانس و انحراف معیار
        cov = torch.matmul(centered, centered.t()) / centered.size(1)
        std = torch.sqrt(torch.sum(centered ** 2, dim=1) / centered.size(1))
        std_matrix = std.unsqueeze(1) * std.unsqueeze(0)

        # محاسبه ماتریس همبستگی
        correlation_matrix = cov / (std_matrix + 1e-8)  # افزودن epsilon برای جلوگیری از تقسیم بر صفر

        # بررسی مقادیر نامعتبر در ماتریس همبستگی
        correlation_matrix = torch.where(
            torch.isnan(correlation_matrix) | torch.isinf(correlation_matrix),
            torch.zeros_like(correlation_matrix),
            correlation_matrix
        )

        # اعمال ماسک مثلثی بالایی (شامل قطر اصلی)
        triu_mask = torch.triu(torch.ones_like(correlation_matrix), diagonal=0).bool()
        correlation_matrix = correlation_matrix * triu_mask

        # ایجاد ماتریس همبستگی کامل با پر کردن صفرها
        full_correlation = torch.zeros(filters.size(0), filters.size(0), device=filters.device, dtype=filters.dtype)
        full_correlation[active_indices[:, None], active_indices] = correlation_matrix

        return full_correlation

    def forward(self, weights, mask):
        correlation_matrix = self.pearson_correlation(weights, mask)
        mask = mask.squeeze(-1).squeeze(-1).squeeze(-1)
        mask_matrix = mask.unsqueeze(1) * mask.unsqueeze(0)
        
        # اعمال عملیات مثلثی بالایی روی mask_matrix
        mask_matrix = mask_matrix * torch.triu(torch.ones_like(mask_matrix), diagonal=0).bool()
        
        # اعمال ماسک برای فیلترهای فعال
        masked_correlation = correlation_matrix * mask_matrix
        
        squared_sum = (masked_correlation ** 2).sum()
        num_active = mask_matrix.sum()  # تعداد جفت‌های فعال در بخش مثلثی بالایی
        
        if num_active > 0:
            normalized_loss = squared_sum / num_active
        else:
            normalized_loss = torch.tensor(0.0, device=weights.device, dtype=weights.dtype)

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
