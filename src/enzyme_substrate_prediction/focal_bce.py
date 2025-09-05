import torch
import torch.nn as nn
import torch.nn.functional as F

class MixedFocalWeightedBCELoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None, lambda_focal=0.5):
        """
        Args:
            alpha (float): balancing factor for focal loss (for class imbalance inside focal loss)
            gamma (float): focusing parameter for focal loss
            pos_weight (float or Tensor, optional): weight for positive class for BCE
            lambda_focal (float): mixing weight between focal loss and weighted BCE
        """
        super(MixedFocalWeightedBCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.lambda_focal = lambda_focal

        # Standard BCEWithLogitsLoss (more stable than manual sigmoid + BCE)
        if pos_weight is not None:
            self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: raw model outputs (before sigmoid), shape (batch_size, ...)
            targets: ground truth labels (0 or 1), shape (batch_size, ...)
        """
        # BCE Loss
        bce = self.bce_loss(logits, targets)

        # Focal Loss
        probas = torch.sigmoid(logits)
        probas = probas.clamp(min=1e-6, max=1-1e-6)  # avoid log(0)
        
        # focal loss term
        pt = torch.where(targets == 1, probas, 1 - probas)
        focal_weight = (1 - pt) ** self.gamma

        focal_loss = -self.alpha * focal_weight * (targets * torch.log(probas) + (1 - targets) * torch.log(1 - probas))
        focal_loss = focal_loss.mean()

        # Mixed Loss
        loss = self.lambda_focal * focal_loss + (1 - self.lambda_focal) * bce
        return loss
