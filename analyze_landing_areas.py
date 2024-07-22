import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from crop_leaf import get_background_mask

def find_hole_sizes(hair_mask, background_mask):
    # Invert the hair mask with background
    inverted_mask = cv2.bitwise_not(hair_mask | cv2.bitwise_not(background_mask))
    
    # Find connected components
    num_labels, labels, comp_stats, centroids = cv2.connectedComponentsWithStats(inverted_mask, connectivity=8)
    print(num_labels)
    print(labels)
    print(comp_stats)
    print(centroids)
    # Exclude the background label (0)
    hole_sizes = comp_stats[1:, cv2.CC_STAT_AREA]
    print("bahhh")
    print(hole_sizes)
    
    # Calculate statistics
    mean_size = np.mean(hole_sizes)
    median_size = np.median(hole_sizes)
    max_size = np.max(hole_sizes)
    min_size = np.min(hole_sizes)
    
    # Print statistics
    print(f'Mean hole size: {mean_size}')
    print(f'Median hole size: {median_size}')
    print(f'Max hole size: {max_size}')
    print(f'Min hole size: {min_size}')
    

    #oneeee
    plt.figure(figsize=(10, 6))
    plt.boxplot(hole_sizes, vert=False, patch_artist=True, 
                boxprops=dict(facecolor='lightgreen', color='black'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'))
    plt.title('Box Plot of Hole Sizes')
    plt.xlabel('Hole Size (number of pixels)')
    plt.grid(True)
    plt.savefig("ahh_boxplot.png")

    #twooo
    plt.figure(figsize=(10, 6))
    plt.hist(hole_sizes, color='lightgreen', edgecolor='black', bins=500)  # Increased number of bins
    plt.title('Distribution of Hole Sizes')
    plt.xlabel('Hole Size (number of pixels)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig("ahh_bin.png")

    #threeee
    plt.figure(figsize=(10, 6))
    log_hole_sizes = np.log1p(hole_sizes)  # Apply log transformation
    plt.hist(log_hole_sizes, color='lightgreen', edgecolor='black', bins=30)
    plt.title('Log-Transformed Distribution of Hole Sizes')
    plt.xlabel('Log of Hole Size (number of pixels)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig("ahh_log.png")

    #fourrrr
    plt.figure(figsize=(10, 6))
    plt.hist(hole_sizes, color='lightgreen', edgecolor='black', bins=30, cumulative=True)
    plt.title('Cumulative Distribution of Hole Sizes')
    plt.xlabel('Hole Size (number of pixels)')
    plt.ylabel('Cumulative Frequency')
    plt.grid(True)
    plt.savefig("ahh_cumulative.png")



    
    # Calculate additional statistics if needed
    mode_size = stats.mode(hole_sizes)[0]
    std_dev_size = np.std(hole_sizes)
    
    # Print additional statistics
    print(f'Mode hole size: {mode_size}')
    print(f'Standard deviation of hole sizes: {std_dev_size}')
    
    # Return hole sizes and statistics
    return {
        'hole_sizes': hole_sizes,
        'mean_size': mean_size,
        'median_size': median_size,
        'max_size': max_size,
        'min_size': min_size,
        'mode_size': mode_size,
        'std_dev_size': std_dev_size
    }

# Example usage
if __name__ == "__main__":
    # Load the hair mask image
    hair_mask = cv2.imread('whole_leaf_masks/reconstructed_mask_070-PI483178_13-93.png', cv2.IMREAD_GRAYSCALE)
    background_mask = get_background_mask("leaves_to_inference/070-PI483178_13-93.png")
    # Check if the image was loaded correctly
    if hair_mask is None:
        print("Error loading image")
        exit()
    
    # Find hole sizes and get statistics
    hole_stats = find_hole_sizes(hair_mask, background_mask)
    print(hole_stats)
