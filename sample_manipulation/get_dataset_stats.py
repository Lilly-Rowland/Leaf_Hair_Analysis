
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import torch

def calculate_mean_std(image_folder):
    # List to hold all pixel values
    pixel_values = []

    # Create a transform to convert images to tensors
    transform = transforms.ToTensor()

    # Loop through all files in the image folder
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Open image file
            img_path = os.path.join(image_folder, filename)
            image = Image.open(img_path).convert('RGB')

            # Convert image to tensor and add its pixel values to the list
            image_tensor = transform(image)
            pixel_values.append(image_tensor)

    # Stack all tensors into one tensor
    all_pixels = torch.stack(pixel_values)

    # Compute mean and std across all images
    mean = all_pixels.mean(dim=[0, 2, 3])
    std = all_pixels.std(dim=[0, 2, 3])

    return mean.numpy(), std.numpy()

# Define the path to your image folder
image_folder = 'training_images'

# Calculate mean and std
mean, std = calculate_mean_std(image_folder)

print(f'Mean: {mean}')
print(f'Standard Deviation: {std}')
