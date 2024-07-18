import numpy as np
from PIL import Image
import os

# Function to split RGB image into separate channels
def split_rgb(image):
    r, g, b = image.split()
    return r, g, b

# Directory path for input images
input_dir = "/home/lgr4641/Desktop/Leaf_Hair_Analysis/leaves_to_inference"

# Directory path for output channels
output_dir = "/home/lgr4641/Desktop/Leaf_Hair_Analysis/RGB_channels"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each image in the input directory
for img_name in os.listdir(input_dir):
    # Open image and convert to RGB mode if it's not already
    img_path = os.path.join(input_dir, img_name)
    image = Image.open(img_path).convert('RGB')

    # Split image into RGB channels
    r, g, b = split_rgb(image)

    # Save each channel separately
    r.save(os.path.join(output_dir, f"r_{img_name}"))
    g.save(os.path.join(output_dir, f"g_{img_name}"))
    b.save(os.path.join(output_dir, f"b_{img_name}"))

    print(f"RGB channels saved for {img_name}")
