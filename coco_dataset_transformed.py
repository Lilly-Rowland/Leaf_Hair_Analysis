import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import v2
from pycocotools.coco import COCO

class CocoDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            ann_file (string): Path to the COCO annotation file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]

        # Load image
        img_path = os.path.join(self.img_dir, img_info['path'])
        image = Image.open(img_path).convert("RGB")

        # Create empty mask
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

        # Create mask from annotations
        if anns:
            for ann in anns:
                mask = np.maximum(self.coco.annToMask(ann) * ann['category_id'], mask)

        mask = np.clip(mask, 0, 1)

        # Apply transformations
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, torch.as_tensor(mask, dtype=torch.long)

def apply_transforms(image, mask):
    # Define transformations
    transform = v2.Compose([
        v2.RandomHorizontalFlip(),  # Apply horizontal flipping with 50% probability
        v2.RandomVerticalFlip(),    # Apply vertical flipping with 50% probability
        # Add other transforms as needed (e.g., resizing, normalization)
    ])
    
    # Apply the same transformation to both image and mask
    image, mask = transform(image, mask)
    
    # Convert to tensor and normalize image
    image = T.ToTensor()(image)
    image = T.Normalize(mean=[0.5380782065015497, 0.6146645541178255, 0.4624397479931463], 
                        std=[0.12672495205043693, 0.12178723849002748, 0.1999076104405415])(image)
    
    mask = torch.as_tensor(np.array(mask), dtype=torch.long)
    
    return image, mask

