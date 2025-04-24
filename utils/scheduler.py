import torch.optim.lr_scheduler as lr_scheduler
import math

class CosineAnnealingLRWarmup:
    def __init__(self, optimizer, T_max, eta_min=0, warmup_steps=0, warmup_start_lr=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.current_step = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.cosine_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=eta_min)

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps and self.warmup_steps > 0:
            # Warmup phase
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                fraction = self.current_step / self.warmup_steps
                lr = self.warmup_start_lr + fraction * (base_lr - self.warmup_start_lr)
                param_group['lr'] = lr
        else:
            # Cosine annealing phase
            self.cosine_scheduler.step(self.current_step - self.warmup_steps)

    def state_dict(self):
        return {
            'current_step': self.current_step,
            'cosine_scheduler': self.cosine_scheduler.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']
        self.cosine_scheduler.load_state_dict(state_dict['cosine_scheduler'])
