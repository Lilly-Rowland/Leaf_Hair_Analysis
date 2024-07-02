"""
1. Must be completed: Take 5000 px image and break into tiles of 224 X 224 (label with row and column position)
2. Convert to HSV file and then normalized hue while looping through files
"""

import os
import cv2
import numpy as np

def convert_rgb_to_hsv(input_image_path):
    # Read the RGB image
    rgb_image = cv2.imread(input_image_path)

    # Check if the image was loaded successfully
    if rgb_image is None:
        print(f"Error: Could not load image at {input_image_path}")
        return

    # Convert the RGB image to HSV
    return cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

def histogram_equalization_in_hsv(image, threshold_low=0, threshold_high=255):
    # Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Split the HSV image into different channels
    h, s, v = cv2.split(hsv)
    
    # Apply thresholds to limit the range of pixel values to be equalized
    v = np.where((v >= threshold_low) & (v <= threshold_high), v, v)
    
    # Perform histogram equalization on the V-channel
    s_eq = cv2.equalizeHist(s)
    v_eq = cv2.equalizeHist(v)
    
    # Merge the equalized V-channel back with the original H and S channels
    hsv_eq = cv2.merge((h, s_eq, v_eq))
    
    # Convert the HSV image back to BGR color space
    bgr_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

    return bgr_eq

def normalize_images_in_hsv(input_folder, output_folder, threshold_low=0, threshold_high=255):
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through all files in the input directory
    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        if os.path.isfile(image_path):
            # Read the image
            image = convert_rgb_to_hsv(image_path)
            
            # Apply histogram equalization in the HSV color space
            normalized_image = histogram_equalization_in_hsv(image, threshold_low, threshold_high)
            
            # Save the normalized image to the output directory
            output_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_path, normalized_image)

# Specify the input and output folders
input_folder = "leaf_samples/224x224/leaf_tiles"
output_folder = 'leaf_samples/224x224/s_and_v_normalized_0_to_200'

# Normalize the images in HSV color space
normalize_images_in_hsv(input_folder, output_folder)
