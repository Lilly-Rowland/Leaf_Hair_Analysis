import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw
import json
import os
import numpy as np

class Leaf_Dataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Load annotations from JSON file
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        # Filter annotations for the "hair" category only
        self.annotations = [ann for ann in annotations['annotations'] if ann['category_id'] == 9]
        
        # Create a mapping from image id to annotations
        self.image_id_to_ann = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_ann:
                self.image_id_to_ann[img_id] = []
            self.image_id_to_ann[img_id].append(ann)
        
    def __len__(self):
        return len(self.image_id_to_ann)
    
    def __getitem__(self, idx):
        # Load image
        img_id = list(self.image_id_to_ann.keys())[idx]
        image_info = self.image_id_to_ann[img_id][0]  # Assuming only one annotation per image
        image_path = os.path.join(self.root_dir, image_info['file_path'])
        image = Image.open(image_path).convert('RGB')

        if img_id in self.image_id_to_ann:
            anns = self.image_id_to_ann[img_id]
            mask = torch.zeros((image.size[1], image.size[0]), dtype=torch.uint8)
            for ann in anns:
                segmentation = ann['segmentation']
                polygon = segmentation[0]  # Taking the first polygon
                mask_img = Image.new('L', (image.size[0], image.size[1]), 0)
                ImageDraw.Draw(mask_img).polygon(polygon, outline=1, fill=1)
                mask = transforms.functional.to_tensor(np.array(mask_img))
        else:
            # If no annotations are available, return an empty mask
            mask = torch.zeros((image.size[1], image.size[0]), dtype=torch.uint8)

        if self.transform:
            image = self.transform(image)
        
        return image, mask
