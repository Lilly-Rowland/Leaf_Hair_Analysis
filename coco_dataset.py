import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
from pycocotools.coco import COCO
import random
import numpy as np
from torchvision import transforms

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
        #mask = Image.fromarray(mask)
        # Apply transformations
        # Apply transformations
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, torch.as_tensor(mask, dtype=torch.long)

def transform(image, mask):
    # Define a series of transformations
    transform_img = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.5380782065015497, 0.6146645541178255, 0.4624397479931463],
                    std=[0.12672495205043693, 0.12178723849002748, 0.1999076104405415]),
    ])

    transform_mask = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip()
    ])

    randomSeed = 201
    # Apply the transformations to the image and mask
    if randomSeed == 0:
        seed = random.randint(0, 2147483647)
    else:
        seed = randomSeed
    random.seed(seed)
    torch.manual_seed(seed)
    image = transform_img(image)

    random.seed(seed)
    torch.manual_seed(seed)
    mask = Image.fromarray(mask)
    mask = transform_mask(mask)
    mask = np.array(mask)
    
    return image, mask

# import random
# import numpy as np
# from torchvision import transforms
# from PIL import Image
# from torch.utils.data import Dataset
# from pycocotools.coco import COCO

# class CocoDataset(Dataset):
#     def __init__(self, img_dir, ann_file, transform=None):
#         self.img_dir = img_dir
#         self.ann_file = ann_file
#         self.transform = transform
#         self.coco = COCO(ann_file)
#         self.ids = list(self.coco.imgs.keys())

#     def __len__(self):
#         return len(self.ids)

#     def __getitem__(self, index):
#         img_id = self.ids[index]
#         ann_ids = self.coco.getAnnIds(imgIds=img_id)
#         anns = self.coco.loadAnns(ann_ids)
#         img_info = self.coco.loadImgs(img_id)[0]
#         img_path = os.path.join(self.img_dir, img_info['path'])
        
#         image = Image.open(img_path).convert('RGB')
#         mask = self._load_mask(anns, img_info['height'], img_info['width'])
        
#         if self.transform is not None:
#             seed = 201#np.random.randint(2147483647)  # Get a random seed
#             random.seed(seed)  # Apply this seed to np.random
#             torch.manual_seed(seed)  # Apply this seed to torch
#             image = self.transform(image)
#             random.seed(seed)  # Apply the same seed to np.random again
#             torch.manual_seed(seed)  # Apply the same seed to torch again
#             mask = self.transform(mask)
        
#         return image, mask

#     def _load_mask(self, anns, height, width):
#         mask = np.zeros((height, width), dtype=np.uint8)
#         for ann in anns:
#             rle = self.coco.annToRLE(ann)
#             m = self.coco.decode(rle)
#             mask[m > 0] = 1
#         return Image.fromarray(mask)

# # Define the transform with horizontal and vertical flips
# transform = transforms.Compose([
#     T.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     T.Normalize(mean=[0.5380782065015497, 0.6146645541178255, 0.4624397479931463],
#                     std=[0.12672495205043693, 0.12178723849002748, 0.1999076104405415]),
#     transforms.ToTensor()
# ])

