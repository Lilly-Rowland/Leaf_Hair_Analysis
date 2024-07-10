import numpy as np
from torchvision import transforms
from PIL import Image
from sample_manipulation.data_prep import Leaf_Dataset  # Replace with your dataset loader

# Define paths to your dataset and annotations


# Instantiate your custom dataset
dataset = Leaf_Dataset(root_dir = root_dir, annotation_file=annotation_file, transform=None)  # Use None transform for raw images

# Initialize lists to store channel-wise means and stds
mean_r, mean_g, mean_b = [], [], []
std_r, std_g, std_b = [], [], []

# Compute mean and std for each image in the dataset
for img, _ in dataset:
    img = np.array(img)  # Convert PIL image to numpy array
    
    # Ensure image is in RGB format
    if img.shape[2] == 4:  # Handle RGBA images
        img = img[:, :, :3]
    
    img = img / 255.0  # Convert to [0, 1] float range
    
    mean_r.append(np.mean(img[:, :, 0]))
    mean_g.append(np.mean(img[:, :, 1]))
    mean_b.append(np.mean(img[:, :, 2]))
    std_r.append(np.std(img[:, :, 0]))
    std_g.append(np.std(img[:, :, 1]))
    std_b.append(np.std(img[:, :, 2]))

# Calculate overall mean and std for the dataset
mean = [np.mean(mean_r), np.mean(mean_g), np.mean(mean_b)]
std = [np.mean(std_r), np.mean(std_g), np.mean(std_b)]

print(f"Mean: {mean}")
print(f"Std: {std}")
