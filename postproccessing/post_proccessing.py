import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import math
import cv2
import time

def count_circles_per_hole(landing_area_mask, circle_diameter, contours):
    circle_radius = circle_diameter / 2
    circle_area = np.pi * (circle_radius ** 2)
    
    total_area = 0
    total_circles = 0
    
    # Create an output image to draw the circles
    #output_image = cv2.cvtColor(landing_area_mask, cv2.COLOR_GRAY2BGR)
    circles_per_contour = []
    for contour in contours:
        contour_circles = 0
        # Create a mask for the current contour
        mask = np.zeros_like(landing_area_mask)
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
                       np.all(landing_area_mask[int(center[1]-circle_radius):int(center[1]+circle_radius), int(center[0]-circle_radius):int(center[0]+circle_radius)] == 255):
                        total_circles += 1
                        contour_circles += 1
                        # Draw the circle on the output image
                        #cv2.circle(output_image, (int(center[0]), int(center[1])), int(circle_radius), (0, 0, 255), 2)
                    circles_per_contour.append(contour_circles)                
                j += circle_diameter
            i += circle_diameter
    
    #cv2.imwrite('fitted_circles.png', output_image)  # Optional: remove this if not needed
    circles_per_contour = [num for num in circles_per_contour if num !=0]
    return total_circles, circles_per_contour

def find_sizes(circle_counts, diameter):
    sizes = []
    circle_area = np.pi * (diameter/2.)**2
    for count in circle_counts:
        sizes.append(count * circle_area)
    return sizes

def calculate_stats(hole_sizes, diameter):
    return {
        f"Num Holes (d={diameter})": len(hole_sizes),
        f"Mean (d={diameter})": np.mean(hole_sizes),
        f"Median (d={diameter})": np.median(hole_sizes),
        f"Maximum (d={diameter})": np.max(hole_sizes),
        f"Minimum (d={diameter})": np.min(hole_sizes),
        f"Standard Deviation (d={diameter})": np.std(hole_sizes),
        f"Skewness (d={diameter})": stats.skew(hole_sizes),
        f"Mode (d={diameter})": stats.mode(hole_sizes)[0],
        f"Q1 (d={diameter})": np.percentile(hole_sizes, 25),
        f"Q2 (d={diameter})": np.percentile(hole_sizes, 75),
    }

def calculate_filtered_stats(circle_count, total_hair_pixels, total_leaf_pixels, diameter):
    area = (np.pi * ((diameter / 1.2) / 2.) ** 2) * circle_count
    unfiltered_landing_area = total_leaf_pixels - total_hair_pixels

    return {f"Circle Count (d={diameter})": circle_count,
     f"Landing Area % (d={diameter})": area / total_leaf_pixels,
     f"Filtered Landing Area / Unfiltered Landing Area (d={diameter})": area / unfiltered_landing_area}

def calculate_unfiltered_stats(total_hair_pixels, total_leaf_pixels):
    return {f"Landing Area % (d=n/a)": (total_leaf_pixels-total_hair_pixels) / total_leaf_pixels,
     f"Leaf Hair % (d=n/a)": total_hair_pixels / total_leaf_pixels}

def analyze_landing_areas(landing_area_mask, total_hair_pixels, total_leaf_pixels):
    
    # Get raw holes sizes
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(landing_area_mask, connectivity=8)
    
    raw_hole_sizes = stats[1:, cv2.CC_STAT_AREA]
    
    # Remove holes smaller than min size
    min_size = 8
    hole_sizes = raw_hole_sizes[raw_hole_sizes >= min_size]

    circle_diameters = [4, 15, 30]
    
    # List to store the count of circles fitting into each contour
    contours, _ = cv2.findContours(landing_area_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get circle counts for filtered masks
    circle_count_1, circles_per_contour_1 = count_circles_per_hole(landing_area_mask, circle_diameters[0]/1.2, contours)
    circle_count_2, circles_per_contour_2 = count_circles_per_hole(landing_area_mask, circle_diameters[1]/1.2, contours)
    circle_count_3, circles_per_contour_3 = count_circles_per_hole(landing_area_mask, circle_diameters[2]/1.2, contours)

    # Find filtered hole sizes
    sizes_1 = find_sizes(circles_per_contour_1, circle_diameters[0])
    sizes_2 = find_sizes(circles_per_contour_2, circle_diameters[1])
    sizes_3 = find_sizes(circles_per_contour_3, circle_diameters[2])

    result_data = {}

    result_data.update(calculate_unfiltered_stats(total_hair_pixels, total_leaf_pixels))
    result_data.update(calculate_stats(hole_sizes, "n/a"))

    result_data.update(calculate_filtered_stats(circle_count_1, total_hair_pixels, total_leaf_pixels, circle_diameters[0]))
    result_data.update(calculate_stats(sizes_1, f"{circle_diameters[0]} uM"))

    result_data.update(calculate_filtered_stats(circle_count_2, total_hair_pixels, total_leaf_pixels, circle_diameters[1]))
    result_data.update(calculate_stats(sizes_2, f"{circle_diameters[1]} uM"))

    result_data.update(calculate_filtered_stats(circle_count_3, total_hair_pixels, total_leaf_pixels, circle_diameters[2]))
    result_data.update(calculate_stats(sizes_3, f"{circle_diameters[2]} uM"))

    return result_data

if __name__ == "__main__":
    start_time = time.time()
    # Load the hair mask image
    landing_area_mask = cv2.imread('whole_leaf_masks/inverted_mask_087-PI588540_16-45.png', cv2.IMREAD_GRAYSCALE)
    
    # Random values for total leaf hair pixels and total landing pixel for testing
    stats_results = analyze_landing_areas(landing_area_mask, 10000, 20000)
    print(stats_results)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")