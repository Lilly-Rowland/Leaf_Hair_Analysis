# import os
# import json
# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
# from PIL import Image
# import numpy as np

# class CocoDataset(Dataset):
#     def __init__(self, img_dir, ann_file, transform=None):
#         self.img_dir = img_dir
#         self.transform = transform
#         with open(ann_file) as f:
#             self.annotations = json.load(f)
#         self.imgs = list(sorted(self.annotations["images"], key=lambda x: x['id']))
#         self.anns = {ann["image_id"]: ann for ann in self.annotations["annotations"]}

#     def __len__(self):
#         return len(self.imgs)

#     def __getitem__(self, idx):
#         img_info = self.imgs[idx]
#         img_id = img_info['id']
#         print(img_info)
#         img_path = os.path.join(self.img_dir, img_info['path'])
#         image = Image.open(img_path).convert("RGB")
        
#         ann = self.anns[img_id]
#         mask = Image.new("L", (image.width, image.height))
#         for seg in ann['segmentation']:
#             Image.Draw.Draw(mask).polygon(seg, outline=1, fill=1)
#         mask = np.array(mask, dtype=np.uint8)

#         if self.transform:
#             image = self.transform(image)
#             mask = torch.tensor(mask, dtype=torch.long)

#         return image, mask

# # Define data transformations
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5380782065015497, 0.6146645541178255, 0.4624397479931463], std=[0.12672495205043693, 0.12178723849002748, 0.1999076104405415]),
# ])

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
    # Define any additional transformations you need here
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])
    
    image = transform(image)
    mask = F.resize(Image.fromarray(mask), (256, 256), interpolation=F.InterpolationMode.NEAREST)
    mask = torch.as_tensor(np.array(mask), dtype=torch.long)
    
    return image, mask
