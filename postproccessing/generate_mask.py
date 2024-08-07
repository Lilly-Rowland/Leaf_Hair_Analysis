import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from unet import UNet  # Ensure this is the correct import path for your UNet model
from archs.nested_unet import NestedUNet
from archs.deeplabv3 import DeepLabV3
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
from archs.segnet import SegNet

# Make ground truth mask
def generate_ground_truth_mask(annotation_file, file_name, image_size):
    #print(image_id)
    img_id = 0
    coco = COCO(annotation_file)
    for image in coco.dataset["images"]:
        if image["file_name"] == file_name:
            img_id = image["id"]
            break
    if img_id != 0:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
    else:
        print("Unlabelled image used to generate ground truth mask")

    mask = Image.new('L', image_size, 0)  # Create a blank mask
    draw = ImageDraw.Draw(mask)
    
    for ann in anns:
        if ann['iscrowd'] == 0:  # Only process non-crowd annotations
            segmentation = ann['segmentation']
            if isinstance(segmentation, list):
                for seg in segmentation:
                    poly = np.array(seg).reshape((len(seg) // 2, 2))
                    draw.polygon([tuple(p) for p in poly], outline=1, fill=1)
            else:
                mask_array = coco.annToMask(ann)
                mask_array = Image.fromarray(mask_array)
                mask = Image.composite(mask_array, mask, mask_array)
    return mask

# Load the trained model
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

# Preprocess the input image
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Post-process the oxutput mask
def postprocess_mask(mask, isXE, threshold=0.5):
    mask = mask.squeeze().cpu()  # Remove batch dimension and move to CPU
    
    if isXE:
        mask = torch.argmax(mask, dim=0)  # Convert to class predictions
    else:
        mask = (mask > threshold).type(torch.uint8)  # Convert to binary mask

    mask = mask.numpy()
    return mask

# Generate a mask for a new image
def generate_mask(model, image_path, transform, device, isXE):
    image = preprocess_image(image_path, transform).to(device)
    with torch.no_grad():
        output = model(image)
    mask = postprocess_mask(output, isXE)
    return mask


# Generate masks for all images in a folder
def generate_masks_for_folder(model, folder_path, transform, device, isXE, annotation_file):
    image_files = os.listdir(folder_path)
    for image_file in image_files:
        if image_file.endswith(".png") or image_file.endswith(".jpg"):  # Assuming all images are PNG or JPG
            image_path = os.path.join(folder_path, image_file)
            mask = generate_mask(model, image_path, transform, device, isXE)
            image_id = image_file  # Assuming image file names are their IDs
            ground_truth_mask = generate_ground_truth_mask(annotation_file, image_id, Image.open(image_path).size)
            visualize_mask(image_path, mask, ground_truth_mask)

# Visualize the mask
def visualize_mask(image_path, generated_mask, ground_truth_mask):
    image = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(ground_truth_mask, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title("Generated Mask")
    plt.imshow(generated_mask, cmap='gray')  # Ensure mask is properly visualized
    plt.axis('off')
    plt.show()

# Main function to load model and generate masks for all images in a folder
if __name__ == "__main__":
    model_path = 'models/labelbox_data_DeepLabV3_dice_balanced_bs_32_seed_201_epoch_38.pth'
    
    folder_path = 'tester_images'  # Folder containing multiple images
    annotation_file = 'Data/combined_coco.json'  # Path to your COCO annotation file
    arch = "deeplabv3"
    isXE = False
    n_classes = 1  # Number of classes in your segmentation task
    if isXE:
        n_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Define the same transforms used during training
    # transform = T.Compose([
    #     T.Resize((256, 256)),
    #     T.ToTensor(),
    #     T.Normalize(mean=[0.5380782065015497, 0.6146645541178255, 0.4624397479931463],
    #                 std=[0.12672495205043693, 0.12178723849002748, 0.1999076104405415]),
    # ])
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.35860088, 0.4009117,  0.32194334],
                    std=[0.18724611, 0.19575961, 0.23898095])
    ])

    model = load_model(arch, model_path, n_classes).to(device)
    generate_masks_for_folder(model, folder_path, transform, device, isXE, annotation_file)
