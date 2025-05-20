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



from torch.cuda.amp import autocast
import logging

from torch import amp
import logging
import torch
import torch.nn as nn

# تنظیم لاگر
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()

    def pearson_correlation(self, filters, mask):
        if filters.size(0) != mask.size(0):
            raise ValueError(f"Filters and mask must have same number of channels: {filters.size(0)} vs {mask.size(0)}")
        
        mask = mask.squeeze(-1).squeeze(-1).squeeze(-1)
        if mask.dim() != 1:
            raise ValueError(f"Mask must be squeezable to 1D, got shape: {mask.shape}")

        N = filters.size(0)
        flattened_filters = filters.view(N, -1).float()

        if torch.isnan(flattened_filters).any() or torch.isinf(flattened_filters).any():
            logger.warning("NaN or Inf detected in flattened_filters, returning zero correlation matrix")
            return torch.zeros(N, N, device=filters.device, dtype=filters.dtype)

        with amp.autocast('cuda'):
            # استانداردسازی فیلترها
            mean = flattened_filters.mean(dim=1, keepdim=True)
            std = flattened_filters.std(dim=1, keepdim=True, unbiased=False)
            std = torch.where(std == 0, torch.tensor(1.0, device=std.device), std)
            normalized_filters = (flattened_filters - mean) / std

            correlation_matrix = torch.zeros(N, N, device=filters.device, dtype=filters.dtype)
            indices = torch.triu_indices(row=N, col=N, offset=1, device=filters.device)
            i, j = indices[0], indices[1]

            # محاسبه همبستگی برای هر جفت (i, j)
            for idx in range(len(i)):
                w1, w2 = normalized_filters[i[idx]], normalized_filters[j[idx]]
                # محاسبه ضریب همبستگی پیرسون
                corr = torch.dot(w1, w2) / flattened_filters.size(1)
                correlation_matrix[i[idx], j[idx]] = corr

            if torch.isnan(correlation_matrix).any() or torch.isinf(correlation_matrix).any():
                logger.warning("NaN or Inf detected in correlation_matrix, returning zero matrix")
                return torch.zeros(N, N, device=filters.device, dtype=filters.dtype)

        return correlation_matrix

    def forward(self, weights, mask):
        if weights.device != mask.device:
            mask = mask.to(weights.device)
        if weights.dtype != mask.dtype:
            mask = mask.to(dtype=weights.dtype)

        with amp.autocast('cuda'):
            correlation_matrix = self.pearson_correlation(weights, mask)
            mask = mask.squeeze(-1).squeeze(-1).squeeze(-1)
            
            # ساخت ماتریس ماسک کامل
            mask_matrix = mask.unsqueeze(1) * mask.unsqueeze(0)
            
            # استخراج بخش بالا مثلثی ماتریس ماسک
            mask_upper_tri = torch.triu(mask_matrix, diagonal=1)
            
            # ضرب نظیر به نظیر
            masked_upper_tri = correlation_matrix * mask_upper_tri
            
            # محاسبه مجموع مربعات
            squared_sum = (masked_upper_tri ** 2).sum()
            
            # محاسبه تعداد عناصر فعال
            num_active = mask_upper_tri.sum()
            
            # نرمال‌سازی و مدیریت مورد بدون عنصر فعال
            if num_active > 0:
                normalized_loss = squared_sum / num_active
            else:
                logger.warning(
                    "No active elements in mask_upper_tri (num_active=0). "
                    f"Mask shape: {mask.shape}, Mask values: {mask.tolist()}"
                )
                normalized_loss = torch.tensor(0.0, device=weights.device, dtype=weights.dtype)
            
            # بررسی مقادیر نامعتبر
            if torch.isnan(normalized_loss) or torch.isinf(normalized_loss):
                logger.warning("NaN or Inf detected in normalized_loss, returning 0.0")
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
