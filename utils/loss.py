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
        # بررسی ابعاد ورودی‌ها
        if filters.size(0) != mask.size(0):
            raise ValueError(f"Filters and mask must have same number of channels: {filters.size(0)} vs {mask.size(0)}")
        
        # فشرده‌سازی ماسک به یک بعد
        mask = mask.squeeze(-1).squeeze(-1).squeeze(-1)
        if mask.dim() != 1:
            raise ValueError(f"Mask must be squeezable to 1D, got shape: {mask.shape}")

        # یافتن فیلترهای فعال
        active_indices = torch.where(mask > 0)[0]
        if len(active_indices) == 0:
            print("Warning: No active filters found (all mask values <= 0)")
            return torch.zeros(filters.size(0), filters.size(0), device=filters.device)
        elif len(active_indices Oversized Warning: The following operation would result in a tensor with too many dimensions: [64, 1, 1, 1, 1, 1]

        active_filters = filters[active_indices]
        flattened_filters = active_filters.view(active_filters.size(0), -1)

        # بررسی مقادیر نامعتبر در فیلترها
        if torch.isnan(flattened_filters).any() or torch.isinf(flattened_filters).any():
            print("Warning: NaN or Inf found in flattened filters")
            return torch.zeros(filters.size(0), filters.size(0), device=filters.device)

        # محاسبه ماتریس همبستگی
        try:
            correlation_matrix = torch.corrcoef(flattened_filters)
        except RuntimeError as e:
            print(f"Error in corrcoef: {e}")
            return torch.zeros(filters.size(0), filters.size(0), device=filters.device)

        if correlation_matrix.dim() == 0:
            correlation_matrix = correlation_matrix.view(1, 1)

        # بررسی مقادیر نامعتبر در ماتریس همبستگی
        if torch.isnan(correlation_matrix).any() or torch.isinf(correlation_matrix).any():
            print("Warning: NaN or Inf found in correlation matrix")
            return torch.zeros(filters.size(0), filters.size(0), device=filters.device)

        full_correlation = torch.zeros(filters.size(0), filters.size(0), device=filters.device)
        full_correlation[active_indices[:, None], active_indices] = correlation_matrix
        return full_correlation

    def forward(self, weights, mask):
        # محاسبه ماتریس همبستگی
        correlation_matrix = self.pearson_correlation(weights, mask)

        # ساخت ماتریس ماسک
        mask = mask.squeeze(-1).squeeze(-1).squeeze(-1)
        mask_matrix = mask.unsqueeze(1) * mask.unsqueeze(0)

        # اعمال ماسک روی ماتریس همبستگی
        masked_correlation = correlation_matrix * mask_matrix

        # محاسبه نرم فروبنیوس
        frobenius_norm = torch.norm(masked_correlation, p='fro')

        # بررسی مقادیر نامعتبر در خروجی
        if torch.isnan(frobenius_norm) or torch.isinf(frobenius_norm):
            print("Warning: NaN or Inf found in frobenius norm")
            return torch.tensor(0.0, device=weights.device)

        # چاپ مقادیر برای دیباگ
        print(f"MaskLoss: Frobenius norm = {frobenius_norm.item()}, Active filters = {torch.where(mask > 0)[0].size(0)}")

        return frobenius_norm

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
