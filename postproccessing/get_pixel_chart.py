import cv2
from postproccessing.crop_leaf import get_background_mask
import numpy as np

def process_image(input_image_path, output_image_path):
    # Load the image in grayscale
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Ensure the image is binary (black and white)
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Count the white and black pixels
    white_pixels = np.sum(binary_img == 255)
    black_pixels = np.sum(binary_img == 0) - np.count_nonzero(get_background_mask("leaves_to_inference/093-PI588372_16-59.png")==0)
    
    # Get the original image dimensions
    height, width = img.shape
    
    # Create a new image with white pixels on top and black pixels below
    new_img = (np.ones((height, width), dtype=np.uint8))*100
    
    # Calculate the number of rows for white and black pixel sections
    num_white_rows = white_pixels // width
    num_black_rows = black_pixels // width
    
    # Place the white pixels in the top section of the new image
    new_img[:num_white_rows, :] = 255
    
    # Place the black pixels in the bottom section of the new image
    new_img[num_white_rows:num_white_rows + num_black_rows, :] = 0
    
    # Save the resulting image
    cv2.imwrite(output_image_path, new_img)

# Example usage
input_image_path = 'whole_leaf_masks/reconstructed_mask_093-PI588372_16-59.png'   # Path to the input black and white image
output_image_path = 'output_image.png' # Path to save the processed image
process_image(input_image_path, output_image_path)
