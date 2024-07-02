import os
import cv2
import numpy as np

def compute_global_histogram(input_folder):
    global_hist = np.zeros(256)
    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        if os.path.isfile(image_path):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            hist, _ = np.histogram(image.flatten(), 256, [0,256])
            global_hist += hist
    return global_hist

def compute_global_cdf(global_hist):
    cdf = global_hist.cumsum()
    cdf_normalized = cdf * 255 / cdf.max()  # Normalize CDF to span 0-255
    cdf_normalized = cdf_normalized.astype('uint8')  # Convert to uint8
    return cdf_normalized

def normalize_image(image, global_cdf):
    normalized_image = global_cdf[image]
    return normalized_image

def normalize_images(input_folder, output_folder):
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Compute the global histogram
    global_hist = compute_global_histogram(input_folder)
    
    # Compute the global CDF from the histogram
    global_cdf = compute_global_cdf(global_hist)
    
    # Iterate through all files in the input directory
    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        if os.path.isfile(image_path):
            # Read the image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # Perform histogram normalization using global CDF
            normalized_image = normalize_image(image, global_cdf)
            
            # Save the normalized image to the output directory
            output_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_path, normalized_image)

# Specify the input and output folders
input_folder = 'leaf_samples/blocked_images'
output_folder = 'leaf_samples/normalized_images'

# Normalize the images
normalize_images(input_folder, output_folder)
