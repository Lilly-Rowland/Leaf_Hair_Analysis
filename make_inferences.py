import torch
import os
import shutil
from preprocessing.sample_manipulation.create_image_tiles_by_leaf_id import split_image_into_tiles
import torchvision.transforms as T
from PIL import Image
from unet import UNet  # Import your UNet model class
from archs.nested_unet import NestedUNet
from archs.deeplabv3 import DeepLabV3
from archs.segnet import SegNet
import numpy as np
import pandas as pd
from postproccessing.crop_leaf import get_background_mask
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from postproccessing.post_proccessing import analyze_landing_areas
import time
import logging 

# Dictionary mapping string keys to loss types
LossTypes = {
    'dice': 1,
    'xe': 2,
    'dicebce': 2,
}

def load_model(arch, model_path, n_classes):
    if arch.lower() == "unet":
        model = UNet(3, n_classes)  # Example: Replace with your UNet model instantiation
    elif arch.lower() == "nested_unet":
        model = NestedUNet(3, n_classes)
    elif arch.lower() == "deeplabv3":
        model = DeepLabV3(num_classes=n_classes)
    elif arch.lower() == "segnet":
        model = SegNet(3, n_classes)
    else:
        print("Invalid model")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def calculate_percent_landing_area(mask):
    total_pixels = mask.size
    hair_pixels = np.sum(mask == 1)
    percentage = (hair_pixels / total_pixels) * 100
    return percentage

def create_or_clear_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)  # Remove the directory and all its contents
    
    os.makedirs(directory_path)  # Recreate the directory

# Preprocess the input image
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Post-process the output mask
def postprocess_mask(mask, loss, threshold=0.5):
    mask = mask.squeeze().cpu()  # Remove batch dimension and move to CPU
    
    if loss == "xe":
        mask = torch.argmax(mask, dim=0)  # Convert to class predictions
    else:
        mask = (mask > threshold).type(torch.uint8)  # Convert to binary mask

    mask = mask.numpy()
    return mask

# Generate a mask for a new image
def generate_mask(model, image_path, transform, device, loss):
    image = preprocess_image(image_path, transform).to(device)
    with torch.no_grad():
        output = model(image)
    mask = postprocess_mask(output, loss)
    return mask


def stitch_masks(tile_masks, leaf = "leaf"):
    # Create an empty mask array to combine all masks
    image_size = [5502, 8254]
    stitched_mask = np.zeros(image_size, dtype=np.uint8)

    # Paste each mask into the stitched mask at the correct position
    for item in tile_masks:
        mask = item["mask"]
        x_start = item["col"] * 224  # Calculate start x-coordinate in stitched mask
        y_start = item["row"] * 224  # Calculate start y-coordinate in stitched mask

        # Determine the end coordinates within the stitched mask
        x_end = x_start + 224
        y_end = y_start + 224

        # Skip mask if it is cropped (dimensions do not match expected size)
        if mask.shape[0] != 224 or mask.shape[1] != 224:
            continue
        
        # Overlay the mask onto the stitched mask
        stitched_mask[y_start:y_end, x_start:x_end] = mask
    
    # Clip the values to ensure they are in the valid range for uint8
    stitched_mask = np.clip(stitched_mask*255, 0, 255).astype(np.uint8)

    # Create a PIL Image from the stitched mask array
    stitched_image = Image.fromarray(stitched_mask)

    # Debugging: Visualize stitched_mask before returning
    
    return stitched_image

# def find_hole_sizes():
#     #invert images
#     #find sizes of different obect in image --> make array of the different sizes + do stats
#     # mean, median, max size, min size, distribution, graph of distribution

def main(image_dir, tile_dir, model, loss, results_path, transform, device, make_hair_mask=False):

    print(f"Making inferences on images from {image_dir}")

    mask_dir = f"{os.path.basename(image_dir)}_hair_masks"
    if make_hair_mask:
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        print(f"Leaf hair masks saved to {mask_dir}")


    results_df = pd.DataFrame()
    all_data = []

    create_or_clear_directory(tile_dir)
    logging.info(f"Leaves to be inferenced:\n{os.listdir(image_dir)}")
    for leaf in os.listdir(image_dir):
        logging.info(f"Current leaf: {leaf}")
        start_time = time.time()
        image_path = os.path.join(image_dir, leaf)
        background_mask = get_background_mask(image_path)
        total_leaf_pixels = np.count_nonzero(background_mask)

        if not (leaf.endswith(".png") or leaf.endswith(".jpg")) or leaf.count('_') == 0:
            logging.error(f"{leaf[:-4]} has invalid name")
            continue  # Skip hidden or system directories

        create_or_clear_directory(tile_dir)
        
        split_image_into_tiles(image_path, tile_dir, tile_size=224)
        total_hair_pixels = 0

        tile_masks = []
        tile_paths = []

        # Run model and create reconstructed mask
    
        for tile in os.listdir(tile_dir):
            tile_path = os.path.join(tile_dir, tile)
            mask = generate_mask(model, tile_path, transform, device, loss)
            mask_name = os.path.basename(tile_path)

            row = int(mask_name.split('_')[2])  # Extract row from mask name
            col = int(mask_name.split('_')[3][:-4])  # Extract col from mask name

            tile_masks.append({"mask": mask, "row": row, "col": col})

            total_hair_pixels += np.count_nonzero(mask)

            tile_paths.append(tile_path)


        reconstructed_mask = stitch_masks(tile_masks, leaf).resize((8254, 5502),Image.NEAREST)

        reconstructed_mask = np.array(reconstructed_mask)
   
        reconstructed_mask = reconstructed_mask & background_mask

        landing_area_mask = cv2.bitwise_not(reconstructed_mask | cv2.bitwise_not(background_mask))
        
        mask_stats = analyze_landing_areas(landing_area_mask, total_hair_pixels, total_leaf_pixels)


        if make_hair_mask:
            # Save leaf mask
            reconstructed_mask_image = Image.fromarray(reconstructed_mask)
            reconstructed_mask_image.save(f"{mask_dir}/reconstructed_{leaf}")
            
        end_time = time.time()
        elapsed_time = end_time - start_time

        results = {"Leaf Id": leaf[:-4], "Elapsed Time (sec)": elapsed_time}
        results.update(mask_stats)

        logging.info(f"Finished Inference for {leaf[:-4]} | Time: {elapsed_time}")

        all_data.append(results)

    (all_data)
    logging.info(all_data)

    results_df = pd.DataFrame(all_data)

    if os.path.isfile(results_path):
        os.remove(results_path)
    
    print("Writing to excel")

    results_df.to_excel(results_path, index=False)

    return results_path, mask_dir

def get_inferences(model_path, image_dir, arch, loss, results_folder, make_hair_mask):
    
    logging.basicConfig(filename='inferences.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

    tile_dir = "/tmp/temp_tiles"
    n_classes = LossTypes[loss]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the same transforms used during training
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.35860088, 0.4009117,  0.32194334],
                    std=[0.18724611, 0.19575961, 0.23898095])
    ])

    model = load_model(arch, model_path, n_classes).to(device)

    return main(image_dir, tile_dir, model, n_classes, results_folder, transform, device, make_hair_mask)


if __name__ == "__main__":
    # model_path = 'models/deeplabv3_dice_balanced_bs_32_seed_555_epoch_26.pth'

    # image_dir = "leaves_to_inference"
    # tile_dir = "/tmp/temp_tiles"

    # arch = "deeplabv3"
    # loss = "dice"

    # results = "hair_model_results.xlsx"

    # n_classes = LossTypes[loss]
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Define the same transforms used during training
    # transform = T.Compose([
    #     T.Resize((256, 256)),
    #     T.ToTensor(),
    #     T.Normalize(mean=[0.5380782065015497, 0.6146645541178255, 0.4624397479931463],
    #                 std=[0.12672495205043693, 0.12178723849002748, 0.1999076104405415]),
    # ])

    # model = load_model(arch, model_path, n_classes).to(device)

    # main(image_dir, tile_dir, model, n_classes, results)

    logging.basicConfig(filename='inferences.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

    model_path = 'models/model-2.pth'
    image_dir = "repository06032024_DM_6-8-2024_3dpi_1"
    tile_dir = "/tmp/temp_tiles"

    arch = "deeplabv3"
    loss = "dice"

    results = "hair_model_results.xlsx"

    n_classes = LossTypes[loss]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the same transforms used during training
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.35860088, 0.4009117,  0.32194334],
                    std=[0.18724611, 0.19575961, 0.23898095])
    ])

    model = load_model(arch, model_path, n_classes).to(device)

    main(image_dir, tile_dir, model, n_classes, results, transform, device)
#TO RUN: nohup python -u make_inferences.py > inferences.log &