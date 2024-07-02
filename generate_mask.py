import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from unet_model import UNet  # Ensure this is the correct import path for your UNet model

# Load the trained model
def load_model(model_path, n_classes):
    model = UNet(n_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocess the input image
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Post-process the output mask
def postprocess_mask(mask, threshold=0.5):
    mask = mask.squeeze().cpu()  # Remove batch dimension and move to CPU
    mask = torch.argmax(mask, dim=0)  # Convert to class predictions
    mask = mask.numpy()
    return mask


# Generate a mask for a new image
def generate_mask(model, image_path, transform, device):
    image = preprocess_image(image_path, transform).to(device)
    with torch.no_grad():
        output = model(image)
    mask = postprocess_mask(output)
    return mask

# Visualize the mask
def visualize_mask(image_path, mask):
    image = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("Generated Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.show()

# Main function to load model, generate and visualize mask
if __name__ == "__main__":
    model_path = 'models/unet_model_epoch_25.pth'
    image_path = 'tester_images/014-Alexandria_T119_3_0_27.png'  # Path to the new image
    n_classes = 2  # Number of classes in your segmentation task
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the same transforms used during training
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5380782065015497, 0.6146645541178255, 0.4624397479931463], std=[0.12672495205043693, 0.12178723849002748, 0.1999076104405415]),
    ])

    model = load_model(model_path, n_classes).to(device)
    mask = generate_mask(model, image_path, transform, device)
    visualize_mask(image_path, mask)
