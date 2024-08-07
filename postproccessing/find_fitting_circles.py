import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from postproccessing.crop_leaf import get_background_mask

def count_and_draw_fitting_circles(binary_image, circle_diameter):
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    circle_radius = circle_diameter / 2
    circle_area = np.pi * (circle_radius ** 2)
    
    total_area = 0
    total_circles = 0
    
    # Create an output image to draw the circles
    output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    circles_per_contour = []
    for contour in contours:
        # Create a mask for the current contour
        mask = np.zeros_like(binary_image)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        contour_area = cv2.contourArea(contour)
        total_area += contour_area

        # Find bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Fit circles within the bounding box
        i = y
        while i < y + h:
            j = x
            while j < x + w:
                # Check if the circle is within the contour mask
                center = (j + circle_radius, i + circle_radius)
                if (center[0] + circle_radius <= x + w and center[1] + circle_radius <= y + h and
                    mask[int(center[1]), int(center[0])] == 255):
                    # Check if the entire circle fits within the contour
                    if np.all(mask[int(center[1]-circle_radius):int(center[1]+circle_radius), int(center[0]-circle_radius):int(center[0]+circle_radius)] == 255)and \
                       np.all(binary_image[int(center[1]-circle_radius):int(center[1]+circle_radius), int(center[0]-circle_radius):int(center[0]+circle_radius)] == 255):
                        total_circles += 1
                        # Draw the circle on the output image
                        cv2.circle(output_image, (int(center[0]), int(center[1])), int(circle_radius), (0, 0, 255), 3)
                j += circle_diameter
            i += circle_diameter
    
    return total_circles, total_area, output_image

if __name__ == "__main__":
    # Load the hair mask image
    hair_mask = cv2.imread('leaves_to_inference/007-PI588601_15-11.png', cv2.IMREAD_GRAYSCALE)
    background_mask = get_background_mask("whole_leaf_masks/reconstructed_mask_007-PI588601_15-11.png")
    # Check if the image was loaded correctly
    if hair_mask is None:
        print("Error loading image")
        exit()
    inverted_mask = cv2.bitwise_not(hair_mask | cv2.bitwise_not(background_mask))
    circle_diameter = 15
    # Find hole sizes and get statistics
    number_of_circles, total_area, output_image = count_and_draw_fitting_circles(inverted_mask, circle_diameter)
    area_circles = np.pi * (circle_diameter / 2) ** 2 * number_of_circles
    
    print(f"num circles: {number_of_circles}\narea circles: {area_circles}\ntotal_area: {total_area}")
    
    # Display the result
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title('Fitted Circles')
    plt.axis('off')
    plt.show()
    
    # Save the result
    cv2.imwrite('fitted_circles.png', output_image)