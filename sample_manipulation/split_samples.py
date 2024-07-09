import os
import shutil
from math import ceil
import random

def split_images(src_folder, dst_folder, num_folders=10):
    # Create the destination folder if it doesn't exist
    os.makedirs(dst_folder, exist_ok=True)

    # Get list of all images in the source folder
    images = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    total_images = len(images)
    images_per_folder = ceil(total_images / num_folders)

    # Shuffle the images
    random.shuffle(images)

    # Split images and copy to new folders
    for i in range(num_folders):
        # Create a new subfolder in the destination folder
        folder_name = f"sample_{i+1}"
        folder_path = os.path.join(dst_folder, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # Calculate start and end indices for current folder
        start_idx = i * images_per_folder
        end_idx = min(start_idx + images_per_folder, total_images)

        # Copy images to the current folder
        for img in images[start_idx:end_idx]:
            src_path = os.path.join(src_folder, img)
            dst_path = os.path.join(folder_path, img)
            shutil.copy(src_path, dst_path)

if __name__ == "__main__":
    src_folder = "/Volumes/Image Data /Sampled_Tiles"  # Replace with the path to your source folder
    dst_folder = "/Volumes/Image Data /sample_splits"  # Replace with the path to your destination folder

    split_images(src_folder, dst_folder)
