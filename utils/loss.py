import torch
import torch.nn as nn
import torch.nn.functional as F


class KDLoss(nn.Module):
    def __init__(self):
        super(KDLoss, self).__init__()
        self.kd_loss = nn.KLDivLoss(reduction='batchmean')
        self.eps = 1e-8  

    def forward(self, logits_teacher_scaled, logits_student_scaled):
        if logits_student_scaled.dim() == 1:
            logits_student_scaled = logits_student_scaled.unsqueeze(1)
        if logits_teacher_scaled.dim() == 1:
            logits_teacher_scaled = logits_teacher_scaled.unsqueeze(1)

        p_teacher_pos = torch.sigmoid(logits_teacher_scaled)
        p_student_pos = torch.sigmoid(logits_student_scaled)


        p_teacher = torch.cat([p_teacher_pos, 1 - p_teacher_pos + self.eps], dim=1)
        p_student = torch.cat([p_student_pos, 1 - p_student_pos + self.eps], dim=1)

        
        p_teacher = p_teacher / (p_teacher.sum(dim=1, keepdim=True) + self.eps)
        log_p_student = torch.log(p_student + self.eps)

        kl_loss = self.kd_loss(log_p_student, p_teacher)
        return kl_loss

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

    def forward(self, Flops, Flops_baseline, compress_rate):
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
