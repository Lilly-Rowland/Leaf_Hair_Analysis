import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image, ImageCms
import time
from skimage.filters import threshold_otsu

def keep_largest_component(binary_image):

    # Find contours of all objects
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, return original image
    if not contours:
        return binary_image
    
    # Find the largest contour based on area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create a mask for the largest contour
    mask = np.zeros_like(binary_image)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    # Apply the mask to the binary image
    largest_component_image = cv2.bitwise_and(binary_image, mask)
    
    return largest_component_image

def fill_holes(image):

    # Convert to binary
    thresholded_image = image > threshold_otsu(image)

    # Accumulated true from left to right (horizontal)
    trueLeft = np.logical_or.accumulate(thresholded_image, axis=1)

    # Accumulated true from right to left (horizontal)
    trueRight = np.logical_or.accumulate(thresholded_image[:, ::-1], axis=1)[:, ::-1]

    # Accumulated true from top to bottom (vertical)
    trueTop = np.logical_or.accumulate(thresholded_image, axis=0)

    # Accumulated true from bottom to top (vertical)
    trueBottom = np.logical_or.accumulate(thresholded_image[::-1, :], axis=0)[::-1]

    # True if there's any true in both horizontal directions (left and right)
    horizontal_or_image = trueLeft * trueRight

    # True if there's any true in both vertical directions (top and bottom)
    vertical_or_image = trueTop * trueBottom

    # True if there's any true in both horizontal and vertical directions
    filled_image = horizontal_or_image & vertical_or_image

    return filled_image.astype(np.uint8) * 255

def create_mask(binary_image):

    # Set kernel
    kernel = np.ones((25, 25), np.uint8)
    
    # Closing holes
    dilated_image = cv2.dilate(binary_image, kernel, iterations=8, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    eroded_image = cv2.erode(dilated_image, kernel, iterations=8)
  
    # Select for largest connected object
    cleaned_image = keep_largest_component(eroded_image)

    filled_image = fill_holes(cleaned_image)

    # Blur to smooth edges
    blurred_image = cv2.GaussianBlur(filled_image, (15, 15),5)

    return filled_image


def crop_leaf_disc(image_path, justMask = False):
    
    color_threshold = 140

    # Load the image
    im = Image.open(image_path).convert('RGB')
    image = cv2.imread(image_path)

    # Convert to Lab colourspace
    srgb_p = ImageCms.createProfile("sRGB")
    lab_p  = ImageCms.createProfile("LAB")

    rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
    Lab = ImageCms.applyTransform(im, rgb2lab)
    _, _, b_channel = Lab.split()

    b_channel_np = np.array(b_channel)
    _, thresholded_b_channel_np = cv2.threshold(b_channel_np, color_threshold, 255, cv2.THRESH_BINARY)

    # Create mask on thresholded b-channel
    thresholded_b_channel_np = create_mask(thresholded_b_channel_np)
    
    # Resize mask for bitwise and with image
    resized_mask = cv2.resize(thresholded_b_channel_np.astype(np.uint8), (image.shape[1], image.shape[0]))

    if justMask:
        return resized_mask

    resized_mask_3ch = cv2.merge([resized_mask, resized_mask, resized_mask])

    # if justMask:
    #     return resized_mask_3ch
    
    masked_image = cv2.bitwise_and(image, resized_mask_3ch)

    return masked_image

def get_background_mask(image_path):
    mask = crop_leaf_disc(image_path, True)
    return mask

def main():
    # Crop the leaf disc from the images
    for leaf in os.listdir("leaves_to_inference"):
        masked_image = crop_leaf_disc(f"leaves_to_inference/{leaf}")
        cv2.imwrite(f"cropped_leaves/cropped_{os.path.basename(leaf)}", masked_image)
        print(leaf)
if __name__ == "__main__":
    # Record the start time
    start_time = time.time()

    main()

    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
