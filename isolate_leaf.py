# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# import os

# def crop_leaf_disc(image_path, save_path, min_radius, max_radius):
#     # Load the image
#     image = cv2.imread(image_path)

#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#     # resize
#     scaling_factor = 1
#     new_width = int(gray.shape[1] * scaling_factor)
#     new_height = int(gray.shape[0] * scaling_factor)
#     resized_gray = cv2.resize(gray, (new_width, new_height))

#     # Blur the image to reduce noise
#     gray_blurred = cv2.medianBlur(resized_gray, 5)

#     # Apply Hough Circle Transform to find the circles
#     circles = cv2.HoughCircles(
#         gray_blurred, 
#         cv2.HOUGH_GRADIENT, dp=1.2, minDist=new_height,
#         param1=100, param2=30, minRadius=min_radius, maxRadius=max_radius  # Specify radius range here
#     )

#     # Ensure at least some circles were found
#     if circles is not None:
#         circles = np.round(circles[0, :]).astype("int")
#         for (x, y, r) in circles:
#             print(f"Detected circle with center at ({x}, {y}) and radius {r}")
#             # Create a mask with the same dimensions as the image
#             mask = np.zeros_like(image)

#             # Draw a filled circle on the mask
#             cv2.circle(mask, (x, y), r, (255, 255, 255), -1)

#             # Apply the mask to the image to retain only the circular region
#             cropped_image = cv2.bitwise_and(image, mask)

#             # Crop the region of interest
#             x1, y1 = max(0, x - r), max(0, y - r)
#             x2, y2 = min(image.shape[1], x + r), min(image.shape[0], y + r)
#             cropped_image = cropped_image[y1:y2, x1:x2]

#             # Save the cropped image
#             cv2.imwrite(save_path, cropped_image)
#             print(f"Cropped image saved to {save_path}")

#             # Display the cropped image
#             plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
#             plt.show()
#             return cropped_image
#     else:
#         print("No circles were found.")
#         return None

# # Paths for the images
# image_path1 = '345-ThompsonSeedless.png'
# save_path1 = 'ahhh.png'
# image_path2 = '329-sConcord_T190_8.png'
# save_path2 = 'ooh.png'

# # Radius range for the circles (adjust these values as needed)
# min_radius = 3500
# max_radius = 3800

# # Crop the leaf disc from the images
# for leaf in os.listdir("leaves_to_inference"):
#     crop_leaf_disc(f"leaves_to_inference/{leaf}", f"cropped_leaves/cropped_{os.path.basename(leaf)}", min_radius, max_radius)

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def crop_leaf_disc(image_path, save_path, min_radius, max_radius, center_x_range=None, center_y_range=None):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize (if needed)
    scaling_factor = 1
    new_width = int(gray.shape[1] * scaling_factor)
    new_height = int(gray.shape[0] * scaling_factor)
    resized_gray = cv2.resize(gray, (new_width, new_height))

    # Blur the image to reduce noise
    gray_blurred = cv2.medianBlur(resized_gray, 5)

    # Apply Hough Circle Transform to find the circles
    circles = cv2.HoughCircles(
        gray_blurred, 
        cv2.HOUGH_GRADIENT, dp=2, minDist=new_height//2,
        param1=100, param2=30, minRadius=min_radius, maxRadius=max_radius  # Adjust parameters here
    )

    # Ensure at least some circles were found
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            print(f"Detected circle with center at ({x}, {y}) and radius {r}")


        # Filter circles based on center range
        if center_x_range and center_y_range:
            circles = [circle for circle in circles if 
                       (center_x_range[0] <= circle[0] <= center_x_range[1]) and 
                       (center_y_range[0] <= circle[1] <= center_y_range[1])]

        # Ensure there's at least one circle left after filtering
        if len(circles) > 0:
            # Find the circle with the largest radius (most obvious circle)
            largest_circle = min(circles, key=lambda x: x[2])
            x, y, r = largest_circle
            print(f"Detected circle with center at ({x}, {y}) and radius {r}")

            # Create a mask with the same dimensions as the image
            mask = np.zeros_like(image)

            # Draw a filled circle on the mask
            cv2.circle(mask, (x, y), r, (255, 255, 255), -1)

            # Apply the mask to the image to retain only the circular region
            cropped_image = cv2.bitwise_and(image, mask)

            # Crop the region of interest
            x1, y1 = max(0, x - r), max(0, y - r)
            x2, y2 = min(image.shape[1], x + r), min(image.shape[0], y + r)
            cropped_image = cropped_image[y1:y2, x1:x2]

            # Save the cropped image
            cv2.imwrite(save_path, cropped_image)
            print(f"Cropped image saved to {save_path}")

            # Display the cropped image
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
min_radius = 2000
max_radius = 4000

# Center ranges (adjust these values based on your image characteristics)
center_x_range = (3500, 4500)
center_y_range = (2000, 3000)

# Crop the leaf disc from the images
for leaf in os.listdir("leaves_to_inference"):
    crop_leaf_disc(f"leaves_to_inference/{leaf}", f"cropped_leaves/cropped_{os.path.basename(leaf)}", min_radius, max_radius, center_x_range, center_y_range)
