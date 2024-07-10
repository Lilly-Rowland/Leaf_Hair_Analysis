import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets):
        assert inputs.shape == targets.shape, "Shapes of inputs and targets must match."
        
        if self.weight is not None:
            weight = self.weight[targets.long()]
            loss = - (weight * (targets * torch.log(inputs + 1e-12) + (1 - targets) * torch.log(1 - inputs + 1e-12))).mean()
        else:
            loss = torch.mean(-targets * torch.log(inputs + 1e-12) - (1 - targets) * torch.log(1 - inputs + 1e-12))
        
        return loss

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.weight = weight
        self.bce_loss = BinaryCrossEntropyLoss(weight=weight)

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        # Compute BCE loss using the custom BCE loss class
        BCE = self.bce_loss(inputs, targets)

        Dice_BCE = BCE / 2. + dice_loss / 2.

        return Dice_BCE