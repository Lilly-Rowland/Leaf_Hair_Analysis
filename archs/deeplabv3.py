import torch
import torch.nn as nn
import torchvision.models.segmentation as models
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet50_Weights

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=1):
        super(DeepLabV3, self).__init__()
        # Use the appropriate weights enum
        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        self.model = models.deeplabv3_resnet50(weights=weights)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model(x)['out']
