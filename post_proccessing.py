import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from crop_leaf import get_background_mask

def count_and_draw_fitting_circles(binary_image, circle_diameter):
    # Find contours in the binary image
    pixel_diameter = circle_diameter/1.2
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    circle_radius = pixel_diameter / 2
    circle_area = np.pi * (circle_radius ** 2)
    
    total_area = 0
    total_circles = 0
    
    # Create an output image to draw the circles
    output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    
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
                        cv2.circle(output_image, (int(center[0]), int(center[1])), int(circle_radius), (0, 0, 255), 2)
                j += circle_diameter
            i += circle_diameter
    
    {f"# Circles Fit (d={circle_diameter} uM)": total_circles,
    f"Landing Area % (d={circle_diameter} uM)": ,
    }
    return total_circles, total_area, output_image

def count_circles_per_hole(landing_area_mask, circle_diameter):
    #delete alter
    output_image = cv2.cvtColor(landing_area_mask, cv2.COLOR_GRAY2BGR)
    
    # List to store the count of circles fitting into each contour
    contours, _ = cv2.findContours(landing_area_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles_per_contour = []
    
    pixel_radius = circle_diameter/2

    total_circles = 0

    for contour in contours:
        # Create a mask for the current contour
        mask = np.zeros_like(landing_area_mask)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Find bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Initialize the circle count for the current contour
        contour_circle_count = 0
        
        # Fit circles within the bounding box
        i = y
        while i < y + h:
            j = x
            while j < x + w:
                # Check if the circle is within the contour mask
                center = (j + pixel_radius, i + pixel_radius)
                if (center[0] + pixel_radius <= x + w and center[1] + pixel_radius <= y + h and
                    mask[int(center[1]), int(center[0])] == 255):
                    # Check if the entire circle fits within the contour
                    if np.all(mask[int(center[1]-pixel_radius):int(center[1]+pixel_radius), int(center[0]-pixel_radius):int(center[0]+pixel_radius)] == 255) and \
                       np.all(landing_area_mask[int(center[1]-pixel_radius):int(center[1]+pixel_radius), int(center[0]-pixel_radius):int(center[0]+pixel_radius)] == 255):
                        total_circles += 1
                        contour_circle_count += 1
                        # Optional: Draw the circle on the output image
                        cv2.circle(output_image, (int(center[0]), int(center[1])), int(pixel_radius), (0, 0, 255), 2)
                j += pixel_radius*2
            i += pixel_radius*2

        # Append the count of circles for the current contour
        circles_per_contour.append(contour_circle_count)
    
    return total_circles, circles_per_contour

def analyze_landing_area(landing_area_mask):
    circle_diameters = [4, 15]
    filtered_mask_1 = create_filtered_masks(landing_area_mask, circle_diameters[0]/1.2)
    filtered_mask_2 = create_filtered_masks(landing_area_mask, circle_diameters[1]/1.2)

    return {f"# Circles Fit (d={circle_diameters[0]*1.2} uM)": ,
            f"# Circles Fit (d={circle_diameters[1]*1.2} uM)": ,
            f"Landing Area % (d={circle_diameters[0]*1.2} uM)": ,
            f"Landing Area % (d={circle_diameters[1]*1.2} uM)": ,
            f"Mean (d={circle_diameters[0]*1.2} uM)": ,
            f"Mean (d={circle_diameters[0]*1.2} uM)": ,
            f"Diameter {circle_diameters[0]*1.2} uM": ,
            }

if __name__ == "__main__":
    # Load the hair mask image
    landing_area_mask = cv2.imread('whole_leaf_masks/reconstructed_mask_070-PI483178_13-93.png', cv2.IMREAD_GRAYSCALE)
    analyze_landing_area(landing_area_mask=landing_area_mask)
    circle_diameters = [4/1.2, 15/1.2]
    
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

"""
- Take leaf mask
- invert leaf mask
- create filtered masks

"""