import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image, ImageCms
import time

# Record the start time
start_time = time.time()

def interpolate_image(binary_image):
    # Morphological operations to fill small gaps and smooth edges
    kernel = np.ones((31, 31), np.uint8)
    dilated_image = cv2.dilate(binary_image, kernel, iterations=8)
    eroded_image = cv2.erode(dilated_image, kernel, iterations=8)
    blurred_image = cv2.GaussianBlur(eroded_image, (15, 15), 2)
    return blurred_image

def preprocess_image(image):
    blurred_image = cv2.GaussianBlur(image, (15, 15), 2)

    # Increase contrast
    alpha = 2.0  # Contrast control (1.0-3.0)
    beta = 0    # Brightness control (0-100)
    contrasted_image = cv2.convertScaleAbs(blurred_image, alpha=alpha, beta=beta)

    # Apply a high-pass filter to enhance edges
    sharp = cv2.subtract(contrasted_image, blurred_image)
    enhanced_edges_image = cv2.addWeighted(contrasted_image, 1.5, sharp, -0.5, 0)

    # Adaptive Thresholding
    binary_image = cv2.adaptiveThreshold(
        enhanced_edges_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    return binary_image

def crop_leaf_disc(image_path, save_path, min_radius, max_radius, center_x_range=None, center_y_range=None):
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

    #thresholded_b_channel_np = preprocess_image(thresholded_b_channel_np)
    thresholded_b_channel_np = interpolate_image(thresholded_b_channel_np)
    

    # Resize (if needed)
    scaling_factor = 1
    new_width = int(thresholded_b_channel_np.shape[1] * scaling_factor)
    new_height = int(thresholded_b_channel_np.shape[0] * scaling_factor)
    resized_img = cv2.resize(thresholded_b_channel_np, (new_width, new_height))

    # Blur the image to reduce noise
    img_blurred = cv2.medianBlur(resized_img, 9)

    plt.imshow(thresholded_b_channel_np, cmap='gray')
    plt.show()
    cv2.imwrite("ooh.png",thresholded_b_channel_np)
    # Apply Hough Circle Transform to find the circles
    circles = cv2.HoughCircles(
        img_blurred, 
        cv2.HOUGH_GRADIENT, dp=2, minDist=new_height//2,
        param1=100, param2=30, minRadius=min_radius, maxRadius=max_radius
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        if center_x_range and center_y_range:
            circles = [circle for circle in circles if 
                       (center_x_range[0] <= circle[0] <= center_x_range[1]) and 
                       (center_y_range[0] <= circle[1] <= center_y_range[1])]

        if len(circles) > 0:
            largest_circle = max(circles, key=lambda x: x[2])
            x, y, r = largest_circle
            print(f"Detected circle with center at ({x}, {y}) and radius {r}")

            mask = np.zeros_like(image)
            cv2.circle(mask, (x, y), r, (255, 255, 255), -1)

            masked_image = cv2.bitwise_and(image, mask)

            # Calculate bounding box coordinates, allowing for partial cropping
            x1 = max(0, x - r)
            y1 = max(0, y - r)
            x2 = min(image.shape[1], x + r)
            y2 = min(image.shape[0], y + r)

            cropped_image = masked_image[y1:y2, x1:x2]

            cv2.imwrite(save_path, cropped_image)
            print(f"Cropped image saved to {save_path}")

            plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            plt.show()
            return cropped_image
        else:
            print("No circles within the specified center range were found.")
            return None
    else:
        print("No circles were found.")
        return None

# Paths for the images
image_path1 = '345-ThompsonSeedless.png'
save_path1 = 'ahhh.png'
image_path2 = '329-sConcord_T190_8.png'
save_path2 = 'ooh.png'

# Radius range for the circles (adjust these values as needed)
min_radius = 3400
max_radius = 3800

# Center ranges (adjust these values based on your image characteristics)
center_x_range = (3000, 5000)
center_y_range = (1000, 4000)

# Crop the leaf disc from the images
for leaf in os.listdir("leaves_to_inference"):
    crop_leaf_disc(f"leaves_to_inference/{leaf}", f"cropped_leaves_faster/cropped_{os.path.basename(leaf)}", min_radius, max_radius, center_x_range, center_y_range)


end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"Elapsed time: {elapsed_time:.4f} seconds")