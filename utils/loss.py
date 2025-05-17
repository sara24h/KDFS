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
       
        mask = mask.squeeze(-1).squeeze(-1).squeeze(-1) 
        active_indices = torch.where(mask > 0)[0]  
        if len(active_indices) == 0:  
            return torch.zeros(filters.size(0), filters.size(0), device=filters.device)
        
        active_filters = filters[active_indices]  
        flattened_filters = active_filters.view(active_filters.size(0), -1)  
        
    
        correlation_matrix = torch.corrcoef(flattened_filters)
     
        if correlation_matrix.dim() == 0:
            correlation_matrix = correlation_matrix.view(1, 1)
        
       
        full_correlation = torch.zeros(filters.size(0), filters.size(0), device=filters.device)
        full_correlation[active_indices[:, None], active_indices] = correlation_matrix
        return full_correlation

    def forward(self, weights, mask):
     
        correlation_matrix = self.pearson_correlation(weights, mask)  
        
        
        mask = mask.squeeze(-1).squeeze(-1).squeeze(-1)  
        mask_matrix = mask.unsqueeze(1) * mask.unsqueeze(0)  
        
       
        masked_correlation = correlation_matrix * mask_matrix
        
       
        frobenius_norm = torch.norm(masked_correlation, p='fro')
        
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
