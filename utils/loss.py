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
            return torch.zeros(1, 1, device=filters.device, dtype=filters.dtype), active_indices
        elif len(active_indices) == 1:
            return torch.ones(1, 1, device=filters.device, dtype=filters.dtype), active_indices

        # انتخاب فیلترهای فعال و کلیپ کردن
        active_filters = filters[active_indices]
        active_filters = torch.clamp(active_filters, min=-1e10, max=1e10)
        flattened_filters = active_filters.view(active_filters.size(0), -1)

        # بررسی مقادیر نامعتبر (NaN یا Inf)
        if torch.isnan(flattened_filters).any() or torch.isinf(flattened_filters).any():
            return torch.zeros(1, 1, device=filters.device, dtype=filters.dtype), active_indices

        # محاسبه ماتریس همبستگی با torch.corrcoef
        correlation_matrix = torch.corrcoef(flattened_filters)
        correlation_matrix = correlation_matrix.to(dtype=filters.dtype)

        # بررسی مقادیر نامعتبر در ماتریس همبستگی
        correlation_matrix = torch.where(
            torch.isnan(correlation_matrix) | torch.isinf(correlation_matrix),
            torch.zeros_like(correlation_matrix, dtype=filters.dtype),
            correlation_matrix
        )

        # اعمال ماسک مثلثی بالایی
        triu_mask = torch.triu(torch.ones_like(correlation_matrix, dtype=torch.bool), diagonal=0)
        correlation_matrix = correlation_matrix * triu_mask

        return correlation_matrix, active_indices

    def forward(self, weights, mask):
        correlation_matrix, active_indices = self.pearson_correlation(weights, mask)
        mask = mask.squeeze(-1).squeeze(-1).squeeze(-1)
        
        if len(active_indices) <= 1:
            return torch.tensor(0.0, device=weights.device, dtype=weights.dtype)

        # ایجاد mask_matrix با torch.outer
        active_mask = mask[active_indices]
        mask_matrix = torch.outer(active_mask, active_mask)
        mask_matrix = mask_matrix * torch.triu(torch.ones_like(mask_matrix, dtype=torch.bool), diagonal=0)

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
