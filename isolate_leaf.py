# # import cv2
# # import numpy as np
# # from matplotlib import pyplot as plt

# # def isolate_leaf(image_path):
# #     # Load the image
# #     image = cv2.imread(image_path)
# #     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# #     # Convert to grayscale
# #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# #     # Apply GaussianBlur to reduce noise and improve edge detection
# #     blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# #     # Use Canny edge detection to find edges
# #     edges = cv2.Canny(blurred, 50, 150)

# #     # Find contours
# #     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# #     # Create a mask from the largest contour (assuming it's the leaf)
# #     mask = np.zeros_like(gray)

# #     print("Mask unique values:", np.unique(mask))

# #     if contours:
# #         largest_contour = max(contours, key=cv2.contourArea)
# #         cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

# #     # Apply the mask to the original image
# #     leaf_only = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

# #     return image_rgb, mask, leaf_only

# # # Example usage
# # image_path = "345-ThompsonSeedless.png"  # Replace with your actual image path
# # original_image, mask, leaf_image = isolate_leaf(image_path)

# # # Display the results
# # plt.figure(figsize=(15, 5))
# # plt.subplot(1, 3, 1)
# # plt.title('Original Image')
# # plt.imshow(original_image)
# # plt.axis('off')

# # plt.subplot(1, 3, 2)
# # plt.title('Mask')
# # plt.imshow(mask, cmap='gray')
# # plt.axis('off')

# # plt.subplot(1, 3, 3)
# # plt.title('Leaf Only')
# # plt.imshow(leaf_image)
# # plt.axis('off')

# # plt.show()

# # # Calculate the number of non-background pixels (leaf area)
# # leaf_area = np.sum(mask > 0)
# # print(f'The number of non-background (leaf) pixels in the image is: {leaf_area}')


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def detect_leaf_area(image_path):
#     # Load the image
#     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Apply Gaussian blur to smooth the edges
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
#     # Apply binary thresholding to create a mask
#     _, binary_mask = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    
#     # Find contours in the binary mask
#     contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Debug: Check number of contours found
#     print("Number of contours found:", len(contours))
    
#     # Create a blank mask to draw the contours
#     mask = np.zeros_like(binary_mask)
    
#     # Draw the contours on the mask
#     cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    
#     # Debug: Print unique values in the mask
#     print("Mask unique values:", np.unique(mask))
    
#     # Calculate the number of non-background (leaf) pixels
#     num_leaf_pixels = np.sum(mask == 255)
#     print("The number of non-background (leaf) pixels in the image is:", num_leaf_pixels)
    
#     # Display the grayscale image, blurred image, binary mask, and mask with contours
#     plt.figure(figsize=(20, 5))
#     plt.subplot(1, 4, 1)
#     plt.title("Grayscale Image")
#     plt.imshow(gray, cmap='gray')
#     plt.axis('off')

#     plt.subplot(1, 4, 2)
#     plt.title("Blurred Image")
#     plt.imshow(blurred, cmap='gray')
#     plt.axis('off')

#     plt.subplot(1, 4, 3)
#     plt.title("Binary Mask")
#     plt.imshow(binary_mask, cmap='gray')
#     plt.axis('off')

#     plt.subplot(1, 4, 4)
#     plt.title("Leaf Area Mask")
#     plt.imshow(mask, cmap='gray')
#     plt.axis('off')

#     plt.show()

#     return mask

# # Usage
# image_path = '345-ThompsonSeedless.png'
# leaf_mask = detect_leaf_area(image_path)
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_leaf_area(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to smooth the edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize variables for best circle fitting
    best_circle = None
    best_mask = np.zeros_like(gray)
    best_num_leaf_pixels = 0
    
    # Diameter of the circle (1 cm)
    circle_diameter = 1  # in cm, adjust this based on your actual requirement
    
    # Convert circle diameter from cm to pixels (assuming known pixel-per-cm ratio)
    # For example, if 1 cm equals 10 pixels: circle_radius = (circle_diameter / 2) * 10
    circle_radius_pixels = int((circle_diameter / 2) * 10)  # Adjust 10 based on your scale
    
    # Iterate over contours to find the best fitting circle
    for contour in contours:
        # Fit a circle to the contour
        (x, y), radius = cv2.minEnclosingCircle(contour)
        
        # Convert radius to integer (optional)
        radius = int(radius)
        
        # Ensure the fitted circle is within bounds of the image
        if x - radius >= 0 and y - radius >= 0 and x + radius < gray.shape[1] and y + radius < gray.shape[0]:
            # Create a mask for the current circle
            mask = np.zeros_like(gray)
            cv2.circle(mask, (int(x), int(y)), radius, (255), -1)
            
            # Calculate the number of non-background (leaf) pixels
            num_leaf_pixels = np.sum(mask == 255)
            
            # Update best fitting circle based on number of leaf pixels
            if num_leaf_pixels > best_num_leaf_pixels:
                best_num_leaf_pixels = num_leaf_pixels
                best_circle = (x, y, radius)
                best_mask = mask
    
    # Debug: Print number of contours and unique values in the mask
    print("Number of contours found:", len(contours))
    print("Mask unique values:", np.unique(best_mask))
    print("The number of non-background (leaf) pixels in the image is:", best_num_leaf_pixels)
    
    # Display the results for visualization
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(edges, cmap='gray')
    plt.title("Edge-Detected Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(best_mask, cmap='gray')
    plt.title("Leaf Contour Mask (Best Fit Circle)")
    plt.axis('off')
    
    plt.show()

    return best_mask

# Usage
image_path = "345-ThompsonSeedless.png"
leaf_mask = detect_leaf_area(image_path)
