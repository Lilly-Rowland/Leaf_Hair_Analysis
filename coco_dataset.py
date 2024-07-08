import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
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

def transform(image, mask):
    # Define transformations for the image
    transform_image = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5380782065015497, 0.6146645541178255, 0.4624397479931463],
                    std=[0.12672495205043693, 0.12178723849002748, 0.1999076104405415]),
    ])
    
    # Apply transformations to the image
    image = transform_image(image)
    
    # Resize and convert mask to tensor
    transform_mask = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST),
    ])
    
    mask = transform_mask(Image.fromarray(mask))
    mask = torch.as_tensor(np.array(mask), dtype=torch.long)


    return image, mask

