import os
import cv2
import numpy as np

def compute_global_histogram(input_folder, threshold_low, threshold_high):
    global_hist_r = np.zeros(256)
    global_hist_g = np.zeros(256)
    global_hist_b = np.zeros(256)
    
    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        if os.path.isfile(image_path):
            image = cv2.imread(image_path)
            # Split the image into R, G, B channels
            b, g, r = cv2.split(image)
            hist_r, _ = np.histogram(r.flatten(), 256, [0,256])
            hist_g, _ = np.histogram(g.flatten(), 256, [0,256])
            hist_b, _ = np.histogram(b.flatten(), 256, [0,256])
            global_hist_r += hist_r
            global_hist_g += hist_g
            global_hist_b += hist_b
    
    return global_hist_r, global_hist_g, global_hist_b

def compute_global_cdf(global_hist, threshold_low, threshold_high):
    # Apply threshold
    global_hist[:threshold_low] = 0
    global_hist[threshold_high + 1:] = 0
    
    # Compute the cumulative distribution function (CDF)
    cdf = global_hist.cumsum()
    # Normalize CDF to span 0-255
    cdf_normalized = (cdf - cdf[threshold_low]) * 255 / (cdf[threshold_high] - cdf[threshold_low])
    cdf_normalized = np.clip(cdf_normalized, 0, 255)
    cdf_normalized = cdf_normalized.astype('uint8')  # Convert to uint8
    return cdf_normalized

def normalize_image(image, global_cdf_r, global_cdf_g, global_cdf_b, threshold_low, threshold_high):
    # Split the image into R, G, B channels
    b, g, r = cv2.split(image)
    
    # Normalize each channel using the corresponding global CDF and thresholds
    r_normalized = np.where((r >= threshold_low) & (r <= threshold_high), global_cdf_r[r], r)
    g_normalized = np.where((g >= threshold_low) & (g <= threshold_high), global_cdf_g[g], g)
    b_normalized = np.where((b >= threshold_low) & (b <= threshold_high), global_cdf_b[b], b)
    
    # Merge the channels back into a color image
    normalized_image = cv2.merge((b_normalized, g_normalized, r_normalized))
    
    return normalized_image

def normalize_images(input_folder, output_folder, threshold_low=0, threshold_high=255):
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Compute the global histograms for R, G, B channels
    global_hist_r, global_hist_g, global_hist_b = compute_global_histogram(input_folder, threshold_low, threshold_high)
    
    # Compute the global CDFs from the histograms with thresholds
    global_cdf_r = compute_global_cdf(global_hist_r, threshold_low, threshold_high)
    global_cdf_g = compute_global_cdf(global_hist_g, threshold_low, threshold_high)
    global_cdf_b = compute_global_cdf(global_hist_b, threshold_low, threshold_high)
    
    # Iterate through all files in the input directory
    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        if os.path.isfile(image_path):
            # Read the image
            image = cv2.imread(image_path)
            
            # Perform histogram normalization using global CDFs and thresholds
            normalized_image = normalize_image(image, global_cdf_r, global_cdf_g, global_cdf_b, threshold_low, threshold_high)
            
            # Save the normalized image to the output directory
            output_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_path, normalized_image)

# Specify the input and output folders and threshold values
input_folder = 'leaf_samples/blocked_samples'
output_folder = 'leaf_samples/normalized_images/normalized_images_color'
threshold_low = 50   # Lower threshold for normalization
threshold_high = 200 # Upper threshold for normalization

# Normalize the images
normalize_images(input_folder, output_folder, threshold_low, threshold_high)
