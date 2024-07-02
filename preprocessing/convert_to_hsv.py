from PIL import Image
import os
import numpy as np

def rgb_to_hsv(rgb):
    # Normalize the RGB values to the [0, 1] range
    rgb = rgb / 255.0

    # Extract the individual R, G, B channels
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]

    # Compute the max and min values for each pixel
    maxc = np.max(rgb, axis=2)
    minc = np.min(rgb, axis=2)

    # Compute the Value (Brightness)
    v = maxc

    # Compute the Saturation
    s = (maxc - minc) / maxc
    s[maxc == 0] = 0

    # Compute the Hue
    rc = (maxc - r) / (maxc - minc)
    gc = (maxc - g) / (maxc - minc)
    bc = (maxc - b) / (maxc - minc)

    h = np.zeros_like(maxc)
    h[maxc == r] = (bc - gc)[maxc == r]
    h[maxc == g] = 2.0 + (rc - bc)[maxc == g]
    h[maxc == b] = 4.0 + (gc - rc)[maxc == b]
    
    h = (h / 6.0) % 1.0
    h[minc == maxc] = 0.0

    # Scale Hue to [0, 179] and Saturation, Value to [0, 255]
    hsv = np.zeros_like(rgb)
    hsv[:,:,0] = h * 179
    hsv[:,:,1] = s * 255
    hsv[:,:,2] = v * 255

    return hsv.astype('uint8')
directory = "leaf_samples/blocked_images"
for img in os.listdir(directory):
    if "hsv_" + img in os.listdir("leaf_samples/HSV_blocked_samples"):
        continue
    if img == directory:
        continue
    img_path = os.path.join(directory, img)
    # Load the RGB image3
    #image_path = 'leaf_samples/090-Dunkel1xSo_T123_15.png'  # Replace with your image path
    rgb_image = Image.open(img_path).convert('RGB')
    rgb_array = np.array(rgb_image)

    # Convert the RGB image to HSV
    hsv_array = rgb_to_hsv(rgb_array)

    # Convert HSV array to an image
    hsv_image = Image.fromarray(hsv_array, 'RGB')

    # Save the HSV image
    output_path = 'hsv_' + img # Replace with your desired output path
    hsv_image.save("leaf_samples/HSV_blocked_samples/" + output_path)

    # Display the original and HSV images (optional)
    # rgb_image.show()
    # hsv_image.show()
