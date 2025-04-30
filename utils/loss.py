import torch
import torch.nn as nn
import torch.nn.functional as F

class KDLoss(nn.Module):
    def __init__(self):
        super(KDLoss, self).__init__()

    def forward(self, logits_t, logits_s):
        return F.kl_div(
            F.log_softmax(logits_s, dim=1),
            F.softmax(logits_t, dim=1),
            reduction="batchmean",
        )

class RCLoss(nn.Module):
    def __init__(self):
        super(RCLoss, self).__init__()

    @staticmethod
    def rc(x):
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

    def forward(self, x, y):
        if not isinstance(x, list) or not isinstance(y, list):
            raise ValueError("Expected x and y to be lists of feature tensors")
        if len(x) != len(y):
            raise ValueError(f"Feature lists have different lengths: {len(x)} vs {len(y)}")

        loss = 0.0
        for x_i, y_i in zip(x, y):
            if x_i.shape != y_i.shape:
                raise ValueError(f"Feature shapes mismatch: {x_i.shape} vs {y_i.shape}")
            loss += (self.rc(x_i) - self.rc(y_i)).pow(2).mean()
        
        return loss / len(x)

class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()

    def forward(self, Flops, Flops_baseline, compress_rate):
        # Convert to tensors only if not already tensors
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Flops = Flops.clone().detach().to(dtype=torch.float, device=device) if isinstance(Flops, torch.Tensor) else torch.tensor(Flops, dtype=torch.float, device=device)
        Flops_baseline = Flops_baseline.clone().detach().to(dtype=torch.float, device=device) if isinstance(Flops_baseline, torch.Tensor) else torch.tensor(Flops_baseline, dtype=torch.float, device=device)
        compress_rate = compress_rate.clone().detach().to(dtype=torch.float, device=device) if isinstance(compress_rate, torch.Tensor) else torch.tensor(compress_rate, dtype=torch.float, device=device)
        return torch.pow(Flops / Flops_baseline - compress_rate, 2)

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
