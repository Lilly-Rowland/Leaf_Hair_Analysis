import os
import cv2
import numpy as np

def histogram_equalization_in_hsv(image, threshold_low=0, threshold_high=255):
    # Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Split the HSV image into different channels
    h, s, v = cv2.split(hsv)
    
    # Apply thresholds to limit the range of pixel values to be equalized
    v = np.where((v >= threshold_low) & (v <= threshold_high), v, v)
    
    # Perform histogram equalization on the V-channel
    v_eq = cv2.equalizeHist(v)
    
    # Merge the equalized V-channel back with the original H and S channels
    hsv_eq = cv2.merge((h, s, v_eq))
    
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
            image = cv2.imread(image_path)
            
            # Apply histogram equalization in the HSV color space
            normalized_image = histogram_equalization_in_hsv(image, threshold_low, threshold_high)
            
            # Save the normalized image to the output directory
            output_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_path, normalized_image)

# Specify the input and output folders
input_folder = 'leaf_samples/HSV_blocked_samples'
output_folder = 'leaf_samples/normalized_images/normalized_hsv_hue_threshold_200_255'

# Normalize the images in HSV color space
normalize_images_in_hsv(input_folder, output_folder)
