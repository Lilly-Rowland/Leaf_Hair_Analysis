# load the model and get it all set up
# Take leaf disk and separate it into tiles
# generate mask for each tile
# get total landing area %
# calculate holei-ness?

#take file of leafs
#block leafs into 224 x 224 images for each analysis
import torch
import os
import shutil
from sample_manipulation.create_image_tiles_by_leaf_id import split_image_into_tiles
import torchvision.transforms as T
from PIL import Image
from unet import UNet  # Import your UNet model class
from archs.nested_unet import NestedUNet
from archs.deeplabv3 import DeepLabV3
from archs.segnet import SegNet
import numpy as np
import pandas as pd

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


def main(image_dir, tile_dir, model, loss, results):
    columns = ["Leaf Id", "Landing Area %"]
    results_df = pd.DataFrame(columns=columns)
    leaf_pixels = 3850201 #find better way to calculate this?

    create_or_clear_directory(tile_dir)
    count = 0
    for leaf in os.listdir(image_dir):
        if not (leaf.endswith(".png") or leaf.endswith(".jpg")):
            continue  # Skip hidden or system directories
        create_or_clear_directory(tile_dir)
        split_image_into_tiles(os.path.join(image_dir, leaf), tile_dir, tile_size=224)
        total_hair_pixels = 0

        #run model
        for tile in os.listdir(tile_dir):
            tile_path = os.path.join(tile_dir, tile)
            mask = generate_mask(model, tile_path, transform, device, loss)
            total_hair_pixels += np.sum(mask == 1)

        #do more calculations
        print(total_hair_pixels)
        leaf_hair_percent = float(total_hair_pixels)/leaf_pixels
        new_row = {"Leaf Id": leaf[:-4], "Landing Area %": 1 - leaf_hair_percent}
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
        count += 1
        print(count)
        if count > 3:
            break
    print(results_df)
    results_df.to_excel(results, index=False)
            

if __name__ == "__main__":
    model_path = 'models/deeplabv3_dice_balanced_bs_32_seed_555_epoch_26.pth'

    image_dir = "leaves_to_inference"
    tile_dir = "/tmp/temp_tiles"

    arch = "deeplabv3"
    loss = "dice"

    results = "hair_model_results.xlsx"

    n_classes = LossTypes[loss]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the same transforms used during training
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5380782065015497, 0.6146645541178255, 0.4624397479931463],
                    std=[0.12672495205043693, 0.12178723849002748, 0.1999076104405415]),
    ])

    model = load_model(arch, model_path, n_classes).to(device)

    main(image_dir, tile_dir, model, n_classes, results)