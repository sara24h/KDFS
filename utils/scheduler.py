import torch.optim.lr_scheduler as lr_scheduler

class CosineAnnealingLRWarmup:
    def __init__(self, optimizer, T_max, eta_min=0, warmup_steps=0, warmup_start_lr=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.current_step = 0
        self.scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=eta_min)

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.warmup_start_lr + (self.optimizer.param_groups[0]['lr'] - self.warmup_start_lr) * self.current_step / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            self.scheduler.step()
