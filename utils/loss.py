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
        # اطمینان از اینکه تعداد کانال‌ها یکسان است
        if filters.size(0) != mask.size(0):
            raise ValueError(f"Filters and mask must have same number of channels: {filters.size(0)} vs {mask.size(0)}")

        # فشرده‌سازی ماسک به 1D
        mask = mask.squeeze().to(dtype=torch.bool, device=filters.device)
        if mask.dim() != 1:
            raise ValueError(f"Mask must be 1D, got shape: {mask.shape}")

        # انتخاب فیلترهای فعال
        active_indices = torch.nonzero(mask, as_tuple=True)[0]
        n_active = active_indices.size(0)

        # مدیریت موارد خاص (تعداد فیلترهای فعال صفر یا یک)
        if n_active <= 1:
            return torch.zeros(n_active, n_active, device=filters.device, dtype=filters.dtype)

        # انتخاب فیلترهای فعال و فشرده‌سازی به شکل (n_active, -1)
        active_filters = filters[active_indices].view(n_active, -1)

        # بررسی مقادیر نامعتبر یا صفر
        if torch.any(torch.isnan(active_filters)) or torch.any(torch.isinf(active_filters)) or torch.all(active_filters == 0):
            return torch.zeros(n_active, n_active, device=filters.device, dtype=filters.dtype)

        # نرمال‌سازی فیلترها برای محاسبه همبستگی
        norm_filters = F.normalize(active_filters, p=2, dim=1)
        # محاسبه ماتریس همبستگی
        correlation_matrix = torch.matmul(norm_filters, norm_filters.t())

        # بررسی مقادیر نامعتبر در ماتریس همبستگی
        if torch.any(torch.isnan(correlation_matrix)) or torch.any(torch.isinf(correlation_matrix)):
            return torch.zeros(n_active, n_active, device=filters.device, dtype=filters.dtype)

        return correlation_matrix

    def forward(self, weights, mask):
        # محاسبه ماتریس همبستگی
        correlation_matrix = self.pearson_correlation(weights, mask)
        n_active = correlation_matrix.size(0)

        # مدیریت مورد خاص
        if n_active <= 1:
            return torch.tensor(0.0, device=weights.device, dtype=weights.dtype)

        # استخراج مقادیر بالا مثلثی (بدون قطر اصلی)
        triu_indices = torch.triu_indices(row=n_active, col=n_active, offset=1, device=weights.device)
        triu_correlation = correlation_matrix[triu_indices[0], triu_indices[1]]

        # محاسبه مجموع مربعات مقادیر بالا مثلثی
        squared_sum = torch.sum(triu_correlation ** 2)
        num_active = triu_indices.size(1)

        # نرمال‌سازی ضرر
        normalized_loss = squared_sum / num_active if num_active > 0 else torch.tensor(0.0, device=weights.device, dtype=weights.dtype)

        # بررسی مقادیر نامعتبر
        if torch.isnan(normalized_loss) or torch.isinf(normalized_loss):
            return torch.tensor(0.0, device=weights.device, dtype=weights.dtype)

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
