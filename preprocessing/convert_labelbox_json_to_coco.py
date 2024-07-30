import json
import requests
import os
from PIL import Image
import numpy as np
from skimage.draw import polygon
from tqdm import tqdm
import cv2
import re

# Convert numpy types to native Python types
def convert_np_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    return obj

def download_image(url, local_path, headers):
    try:
        response = requests.get(url, stream=True)
        # Print status code and URL for debugging

        if response.status_code == 200:
            with open(local_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
        else:
            print(f"Failed to download {url}. Status code: {response.status_code}. Response text: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

def download_mask(url, local_path, headers):
    try:
        response = requests.get(url, headers=headers, stream=True)

        if response.status_code == 200:
            with open(local_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
        else:
            print(f"Failed to download {url}. Status code: {response.status_code}. Response text: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

def polygon_to_mask(polygon_points, height, width):
    rr, cc = polygon(polygon_points[:, 1], polygon_points[:, 0], shape=(height, width))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[rr, cc] = 1
    return mask

def convert_labelbox_to_coco(labelbox_json_path, coco_json_path, image_dir, headers):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    category_id = 1
    coco_format["categories"].append({
        "id": category_id,
        "name": "Hair"
    })
    
    try:
        with open(labelbox_json_path) as f:
            for line in f:
                try:
                    labelbox_data = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    print(f"JSON Decode Error: {e}")
                    continue

                data_row = labelbox_data.get("data_row", {})
                image_id = data_row.get("id")
                image_url = data_row.get("row_data")
                if not image_url:
                    continue

                image_name = re.search(r'(.*?)(?=\.png)', os.path.basename(image_url)).group(1)
                
                local_image_path = f"{os.path.join(image_dir, image_name)}.png"
                download_image(image_url, local_image_path, headers=headers)

                with Image.open(local_image_path) as img:
                    width, height = img.size

                coco_format["images"].append({
                    "id": image_id,
                    "file_name": image_name,
                    "width": width,
                    "height": height
                })

                labels = labelbox_data.get("projects").get("clxudv7ni0aw9070ybott5mc7").get("labels")
                for label in labels:
                    for obj in label["annotations"]["objects"]:
                        if "composite_mask" in obj:
                            mask_url = f"{obj['composite_mask']['url']}"

                            mask_file_path = "temp_mask.png"
                            download_mask(mask_url, mask_file_path, headers=headers)

                            with Image.open(mask_file_path) as mask_image:
                                mask_image = mask_image.convert('L')
                                mask_array = np.array(mask_image)
                                mask_array = (mask_array > 0).astype(np.uint8) * 255

                                # Find contours in the binary mask image
                                contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                
                                # Convert contours to polygons
                                polygons = []
                                for contour in contours:
                                    # Flatten contour and convert to list
                                    polygon = contour.flatten().tolist()
                                    polygons.append(polygon)
                            

                                coco_format["annotations"].append({
                                    "id": len(coco_format["annotations"]) + 1,
                                    "image_id": image_id,
                                    "category_id": category_id,
                                    "segmentation": polygons,
                                    "area": int(np.sum(mask_array)),
                                    "iscrowd": 0
                                })
                        else:
                            print(f"Unsupported mask format: {mask_url}")

    except FileNotFoundError:
        print(f"File not found: {labelbox_json_path}")

    

    with open(coco_json_path, "w") as f:
        json.dump(coco_format, f, indent=4)

# Usage


labelbox_json_path = 'preprocessing/Export v2 project - Leaf Hair - 7_30_2024.ndjson'
coco_json_path = 'labelbox_coco.json'
image_dir = 'training_images'  # Directory to save downloaded images
api_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbHh1ZHJqdDMwMDgwMDd3YTA1Z3c3eDVtIiwib3JnYW5pemF0aW9uSWQiOiJjazd1dWI3MXUxeW5xMDk4MTZybngzd3d4IiwiYXBpS2V5SWQiOiJjbHo4a2RvMHUwMDFnMDcwODEwMnQ5Mng2Iiwic2VjcmV0IjoiYmQ1YTliNTY3NTM4YzY0ZDdiY2UxMDVlMmJhNjI4YjYiLCJpYXQiOjE3MjIzNTI3ODcsImV4cCI6MjM1MzUwNDc4N30.u1YEVKfGb-btVB0IjPCvb2apYGLdCSIpVT9Cg4Ge7AM'
headers = {
    "Authorization": f"Bearer {api_key}"
}

os.makedirs(image_dir, exist_ok=True)
convert_labelbox_to_coco(labelbox_json_path, coco_json_path, image_dir, headers)



# image_url = "https://storage.labelbox.com/ck7uub71u1ynq09816rnx3wwx%2F40986488-509f-be55-4fda-895c5769ab1a-001-PI588568_13-107_4_21.png"
# mask_url = "https://api.labelbox.com/api/v1/tasks/clz70z72r053y070h46a90121/masks/clxxncipi000400i8nmkf12th/index/1"
# image, mask = fetch_image_and_mask_from_url(image_url, mask_url, api_key)
