import cv2
import numpy as np

def remove_outliers(hole_sizes):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(hole_sizes, 25)
    Q3 = np.percentile(hole_sizes, 75)
    IQR = Q3 - Q1
    
    # Define bounds for outliers
    lower_bound = Q1 - 2.5 * IQR
    upper_bound = Q3 + 2.5 * IQR
    
    # Filter out the outliers
    filtered_hole_sizes = hole_sizes[(hole_sizes >= lower_bound) & (hole_sizes <= upper_bound)]
    
    return filtered_hole_sizes

def remove_small_objects(binary_image_path, min_size, output_image_path):
    # Load the binary image
    binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)
    if binary_image is None:
        raise FileNotFoundError(f"Could not load image at path: {binary_image_path}")

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    # Create an output image initialized to zero (black)
    output_image = np.zeros_like(binary_image)
    
    # Iterate through the connected components and keep only those larger than min_size
    for i in range(1, num_labels):  # Start from 1 to skip the background
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            output_image[labels == i] = 255
    
    # Write the output image
    cv2.imwrite(output_image_path, output_image)
    print(f"Processed image saved as {output_image_path}")

# Example usage
if __name__ == "__main__":
    input_image_path = "mask_included.png"
    output_image_path = "output_binary_image.png"
    min_size = 70  # Minimum size of objects to keep
    
    remove_small_objects(input_image_path, min_size, output_image_path)
