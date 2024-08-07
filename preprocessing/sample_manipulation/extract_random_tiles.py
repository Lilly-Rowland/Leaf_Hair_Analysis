import os
import random
import shutil

def sample_images(source_folder, destination_folder, total_samples):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Count total images in each leaf folder
    leaf_folders = [leaf for leaf in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, leaf))]
    num_folders = len(leaf_folders)
    num_samples_per_folder = total_samples // num_folders
    # Track sampled image counts
    sampled_count = 0
    
    for leaf_folder in leaf_folders:
        leaf_path = os.path.join(source_folder, leaf_folder)
        images = [f for f in os.listdir(leaf_path) if f.endswith('.png')]
        
        # Randomly sample images from the current leaf folder
        sampled_images = random.sample(images, min(num_samples_per_folder, len(images)))
        sampled_count += len(sampled_images)

        # Copy sampled images to the destination folder
        for image in sampled_images:
            src_image_path = os.path.join(leaf_path, image)
            dst_image_path = os.path.join(destination_folder, image)
            shutil.copyfile(src_image_path, dst_image_path)
            
            #print(f"Copied {src_image_path} to {dst_image_path}")
    
    print(f"Total sampled images: {sampled_count}")

# Example usage:
source_folder = '/Volumes/Image Data /Tiled_Repository06032024_DM_ 6-8-2024_3dpi_1'
destination_folder = '/Volumes/Image Data /Sampled_Tiles'
total_samples = 10000

sample_images(source_folder, destination_folder, total_samples)