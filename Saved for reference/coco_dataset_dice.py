import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from pycocotools.coco import COCO
import numpy as np
import torchvision.transforms.functional as TF

class CocoDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        """
        Args:
            img_dir (str): Directory with all the images.
            ann_file (str): Path to the COCO annotation file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.img_ids = list(sorted(self.coco.imgs.keys()))
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

        # Create mask
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in anns:
            mask += self.coco.annToMask(ann)

        # Apply transformations if available
        if self.transform:
            image, mask = self.transform(image, mask)

        # Convert mask to torch.Tensor
        mask = torch.as_tensor(mask, dtype=torch.long)

        return image, mask

# Define a standard transformation function if needed
def transform(image, mask):
    # Example transformation, adjust as needed
    composed_transforms = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])

    image = composed_transforms(image)
    mask = Image.fromarray(mask)
    mask = composed_transforms(mask)

    return image, mask