import os
import cv2
import json
import numpy as np

# Paths to the folders
images_folder = '/Users/lrowland/Desktop/coco-annotator/datasets/leaf'
masks_folder = 'leaf_samples/224x224/s_and_v_normalized_masked'
annotations_file = 'annotations/coco_annotations.json'
binary_images_folder = 'annotations/binary_images'

# Create folders if they don't exist
os.makedirs(binary_images_folder, exist_ok=True)

# Create a list for images and annotations
images = []
annotations = []
categories = [{"id": 1, "name": "hair"}]  # Adjust category name as needed

# Function to create binary image from mask
def create_binary_image(image_path):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define lower and upper HSV thresholds for the pink color range
    lower_threshold = np.array([150, 50, 50])
    upper_threshold = np.array([180, 255, 255])
    
    mask = cv2.inRange(hsv_image, lower_threshold, upper_threshold)
    
    binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    
    return binary_mask

# Iterate over the images
annotation_id = 1
for image_id, image_name in enumerate(os.listdir(images_folder), 1):  # Use images from images_folder
    if image_name.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(images_folder, image_name)
        mask_path = os.path.join(masks_folder, image_name)

        if os.path.exists(mask_path):
            # Add image info to images list
            image = cv2.imread(image_path)
            height, width, _ = image.shape
            images.append({
                "id": image_id,
                "file_name": image_name,
                "width": width,
                "height": height
            })

            # Create binary mask image and save it
            binary_mask = create_binary_image(mask_path)
            cv2.imwrite(os.path.join(binary_images_folder, image_name), binary_mask)

            # Find contours on the binary mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Combine all contours into a single annotation
            segmentation = [contour.flatten().tolist() for contour in contours if len(contour.flatten()) >= 6]

            if segmentation:  # Ensure there are valid segments
                x, y, w, h = cv2.boundingRect(np.vstack(contours))
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": segmentation,
                    "area": np.sum(binary_mask) / 255.0,  # Area of the binary mask
                    "bbox": [x, y, w, h],
                    "iscrowd": 0
                })
                annotation_id += 1
        else:
            print(f"Mask for {image_name} not found.")
    if image_id > 20:
        break

# Create the final annotations dictionary
coco_format = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

# Save the annotations to a JSON file without indentation
with open(annotations_file, 'w') as f:
    json.dump(coco_format, f, separators=(',', ':'))

print(f"Annotations saved to {annotations_file}")
print(f"Binary masks saved to {binary_images_folder}")
