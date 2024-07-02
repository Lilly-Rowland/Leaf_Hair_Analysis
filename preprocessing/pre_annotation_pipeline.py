import os
import cv2
import json
import numpy as np

#Create a list for images and annotations
images = []
annotations = []
categories = [{"id": 1, "name": "hair"}]

#Function to create binary image from mask
def create_binary_image(mask):

    hsv_image = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

    #Define lower and upper HSV thresholds for the pink color range
    lower_threshold = np.array([150, 50, 50])
    upper_threshold = np.array([180, 255, 255])
    
    mask = cv2.inRange(hsv_image, lower_threshold, upper_threshold)
    
    binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    
    return binary_mask

def create_json(image_path, mask, output_path):
    image_name = os.path.basename(image_path)
    image_id = image_name
    annotation_id = image_name

    #Add image info to images list
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    images.append({
        "id": image_id,
        "file_name": image_name,
        "width": width,
        "height": height
    })

    #Create binary mask image and save it
    binary_mask = create_binary_image(mask)

    #Find contours on the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Combine all contours into a single annotation
    segmentation = [contour.flatten().tolist() for contour in contours if len(contour.flatten()) >= 6]

    if segmentation:  #Ensure there are valid segments
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        annotations.append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "segmentation": segmentation,
            "area": np.sum(binary_mask) / 255.0,  #Area of the binary mask
            "bbox": [x, y, w, h],
            "iscrowd": 0
        })
    else:
        print(f"Mask for {image_name} not found.")

    #Create the final annotations dictionary
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    #Save the annotations to a JSON file without indentation
    with open(output_path, 'w') as f:
        json.dump(coco_format, f, separators=(',', ':'))

def convert_rgb_to_hsv(input_image_path):
    #Read the RGB image
    rgb_image = cv2.imread(input_image_path)
    #Check if the image was loaded successfully
    if rgb_image is None:
        print(f"Error: Could not load image at {input_image_path}")
        return
    #Convert the RGB image to HSV
    return cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

def histogram_equalization_in_hsv(image, threshold_low=0, threshold_high=255):
    #Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    #Split the HSV image into different channels
    h, s, v = cv2.split(hsv)
    
    # Apply thresholds to limit the range of pixel values to be equalized
    v = np.where((v >= threshold_low) & (v <= threshold_high), v, v)
    
    #Perform histogram equalization on the V-channel
    s_eq = cv2.equalizeHist(s)
    v_eq = cv2.equalizeHist(v)
    
    #Merge the equalized V-channel back with the original H and S channels
    hsv_eq = cv2.merge((h, s_eq, v_eq))
    
    #Convert the HSV image back to BGR color space
    bgr_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

    return bgr_eq

def normalize_image_in_hsv(image_path, threshold_low=0, threshold_high=255):
    if os.path.isfile(image_path):
        image = convert_rgb_to_hsv(image_path)
        #Apply histogram equalization in the HSV color space
        normalized_image = histogram_equalization_in_hsv(image, threshold_low, threshold_high)
        #Save the normalized image to the output directory
        return normalized_image


def image_to_json(image_path, output_path):
    #Make mask
    mask = normalize_image_in_hsv(image_path)
    #Create JSON fromt the mask
    create_json(image_path, mask, output_path)


image_path = "leaf_samples/224x224/leaf_tiles/009-RemNE11xSo_T139_9_0_2.png"
output_path = "mask.json"

image_to_json(image_path, output_path)

