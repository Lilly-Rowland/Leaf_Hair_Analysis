# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class BinaryCrossEntropyLoss(nn.Module):
#     def __init__(self, weight=None):
#         super(BinaryCrossEntropyLoss, self).__init__()
#         self.weight = weight

#     def forward(self, inputs, targets):
#         assert inputs.shape == targets.shape, "Shapes of inputs and targets must match."
        
#         if self.weight is not None:
#             weight = self.weight[targets.long()]
#             loss = - (weight * (targets * torch.log(inputs + 1e-12) + (1 - targets) * torch.log(1 - inputs + 1e-12))).mean()
#         else:
#             loss = torch.mean(-targets * torch.log(inputs + 1e-12) - (1 - targets) * torch.log(1 - inputs + 1e-12))
        
#         return loss

# class DiceBCELoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceBCELoss, self).__init__()
#         self.weight = weight
#         self.bce_loss = BinaryCrossEntropyLoss(weight=weight)

#     def forward(self, inputs, targets, smooth=1):
#         inputs = torch.sigmoid(inputs)   

#         # Check the shape of targets and make sure it matches inputs
  
#         print(f"Shape of inputs before view: {inputs.shape}")
#         print(f"Shape of targets before view: {targets.shape}")

#         inputs = inputs.view(-1)
#         targets = targets.view(-1)

#         print(f"Shape of inputs after view: {inputs.shape}")
#         print(f"Shape of targets after view: {targets.shape}")

#         intersection = (inputs * targets).sum()
#         dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
#         # Compute BCE loss using the custom BCE loss class
#         BCE = self.bce_loss(inputs, targets)

#         Dice_BCE = BCE / 2. + dice_loss / 2.

#         return Dice_BCE

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
        # Ensure inputs are within (0,1) range
        inputs = torch.sigmoid(inputs)

        # # Check the shapes
        # print("Shape of inputs before view:", inputs.shape)
        # print("Shape of targets before one-hot:", targets.shape)
        
        # Convert targets to one-hot encoding
        num_classes = inputs.shape[1]
        targets = F.one_hot(targets.long(), num_classes).permute(0, 3, 1, 2).float()
        
        # # Check the shapes after one-hot encoding
        # print("Shape of targets after one-hot:", targets.shape)
        
        # Flatten the tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        # # Check the shapes after view
        # print("Shape of inputs after view:", inputs.shape)
        # print("Shape of targets after view:", targets.shape)
        
        # Compute the intersection
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        # Compute BCE loss using the custom BCE loss class
        BCE = self.bce_loss(inputs, targets)

        Dice_BCE = BCE / 2. + dice_loss / 2.

        return Dice_BCE

# Example usage with dummy data
if __name__ == "__main__":
    inputs = torch.randn(2, 2, 224, 224)
    targets = torch.randint(0, 2, (2, 224, 224))

    criterion = DiceBCELoss()
    loss = criterion(inputs, targets)
    print("Loss:", loss.item())
