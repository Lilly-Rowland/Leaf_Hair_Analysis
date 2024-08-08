import torch.nn as nn
import torch.nn.functional as F
import torch

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class WeightedDiceLoss(nn.Module):
    def __init__(self, weight=None):
        super(WeightedDiceLoss, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)  # Assuming outputs are raw logits
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        # Apply weights
        hi = self.weight.numel()
        if self.weight == None or self.weight.numel() == 1.:
            self.weight = torch.ones(2, dtype=torch.float32)
        weighted_dice_loss = (self.weight[1] * dice_loss * targets + self.weight[0] * dice_loss * (1 - targets)).mean()
        
        return weighted_dice_loss