import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError(
                'Reduction {} not implemented.'.format(reduction))
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, target):
        target_cal = target.unsqueeze(1)
        p_t = torch.where(target_cal == 1, x, 1-x)
        alpha_t = torch.where(target_cal == 1, self.alpha, 1-self.alpha)
        fl = - alpha_t * (1 - p_t) ** self.gamma * torch.log(p_t)

        return self._reduce(fl)

    def _reduce(self, x):
        if self.reduction == 'mean':
            return x.mean()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x
