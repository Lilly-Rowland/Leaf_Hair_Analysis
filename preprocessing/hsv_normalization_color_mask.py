import os
import cv2
import numpy as np

def remove_color(image, lower_bounds, upper_bounds):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create masks for each specified color range
    masks = []
    for lower_bound, upper_bound in zip(lower_bounds, upper_bounds):
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        masks.append(mask)
    
    # Combine all masks
    combined_mask = cv2.bitwise_or(masks[0], masks[1])
    for mask in masks[2:]:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Invert the mask to get the colors that are not in the specified ranges
    mask_inv = cv2.bitwise_not(combined_mask)
    
    # Apply the mask to the original image to remove the specified colors
    result = cv2.bitwise_and(image, image, mask=mask_inv)
    
    return result


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
    # Define the HSV ranges for the green color (including lighter yellowish-green)
    lower_bounds = [
    np.array([10, 40, 40]),   # Yellowish-green
    np.array([50, 40, 40]),   # Green
    np.array([70, 40, 40]),    # Cyan-green
    np.array([0, 100, 100]) # Orange
    ]
    upper_bounds = [
    np.array([85, 255, 255]), # Yellowish-green
    np.array([90, 255, 255]), # Green
    np.array([90, 255, 255]),  # Cyan-green
    np.array([20, 255, 255])  # Cyan-green
    ]
    
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

            #Remove green
            masked_image = remove_color(image, lower_bounds, upper_bounds)

            # Save the normalized image to the output directory
            output_path = os.path.join(output_folder, image_name)

            cv2.imwrite(output_path, masked_image)


# Specify the input and output folders
input_folder = "leaf_samples/224x224/leaf_tiles"
output_folder = 'leaf_samples/224x224/s_and_v_normalized_0_to_200'

# Normalize the images in HSV color space
normalize_images_in_hsv(input_folder, output_folder)
