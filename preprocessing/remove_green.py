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

# Define the HSV ranges for the green color (including lighter yellowish-green)
lower_bounds = [
    np.array([10, 40, 40]),   # Yellowish-green
    np.array([50, 40, 40]),   # Green
    np.array([70, 40, 40])    # Cyan-green
]
upper_bounds = [
    np.array([85, 255, 255]), # Yellowish-green
    np.array([90, 255, 255]), # Green
    np.array([90, 255, 255])  # Cyan-green
]

# Load the image
input_image_path = "leaf_samples/224x224/s_and_v_normalized_0_to_200/009-RemNE11xSo_T139_9_18_18.png"
image = cv2.imread(input_image_path)

# Remove the green color from the image
result_image = remove_color(image, lower_bounds, upper_bounds)

# Save or display the result
output_image_path = '/mnt/data/result_no_green.png'
cv2.imwrite(output_image_path, result_image)
cv2.imshow('Original', image)
cv2.imshow('Result', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
