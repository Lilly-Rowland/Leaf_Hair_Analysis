import torch
import torch.optim as optim
import torch.nn as nn
from unet import UNet  # Import your UNet model class
from archs.nested_unet import NestedUNet
from archs.deeplabv3 import DeepLabV3
from archs.segnet import SegNet
from coco_dataset import CocoDataset, transform  # Import your dataset class and transformation function
from torch.utils.data import DataLoader, SubsetRandomSampler
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from losses.dice_loss import DiceLoss, WeightedDiceLoss
from losses.diceBCE_loss import DiceBCELoss
from enum import Enum
import random

# Dictionary mapping string keys to loss types
LossTypes = {
    'dice': 1,
    'xe': 2,
    'dicebce': 2,
}

ArchTypes = ["unet", "nested_unet", "deeplabv3", "segnet"]


def set_random_seed(seed=201):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()

def calculate_class_weights(dataset):
    # Initialize counters
    num_background_pixels = 0
    num_foreground_pixels = 0

    # Loop through the dataset to count pixels
    for _, mask in dataset:
        num_background_pixels += torch.sum(mask == 0).item()
        num_foreground_pixels += torch.sum(mask == 1).item()

    total_pixels = num_background_pixels + num_foreground_pixels
    weight_background = total_pixels / (2 * num_background_pixels)
    weight_foreground = total_pixels / (2 * num_foreground_pixels)

    return torch.tensor([weight_background, weight_foreground])


def train_model(model, loss, train_loader, val_loader, test_dataloader, name, class_weights=None, lr=0.001, gpu_num=2, num_epochs=30, patience=20, tolerance=1e-4):
    device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    if loss.upper() == "DICE":
        if class_weights is None:
            class_weights = torch.ones(1, dtype=torch.float32)
        criterion = WeightedDiceLoss(weight=class_weights)
    elif loss.upper() == "XE":
        if class_weights is None:
            class_weights = torch.ones(2, dtype=torch.float32)
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif loss.upper() == "DICEBCE":
        if class_weights is None:
            class_weights = torch.ones(2, dtype=torch.float32)
        class_weights = class_weights.to(device)
        criterion = DiceBCELoss(weight=class_weights)
    else:
        print("Invalid criterion")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_time = time.time() - start_time
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Time: {epoch_time:.2f} seconds")

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for val_images, val_masks in val_loader:
                val_images = val_images.to(device)
                val_masks = val_masks.to(device)

                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_masks)
                val_running_loss += val_loss.item()
                
        val_loss = val_running_loss / len(val_loader)
        val_losses.append(val_loss)
        #print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

        # Early stopping logic
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            # Save model checkpoint
            torch.save(model.state_dict(), f'models/{name}_epoch_{epoch+1}.pth')
        elif val_loss <= best_val_loss + tolerance:
            best_val_loss = val_loss
            # Save model checkpoint
            torch.save(model.state_dict(), f'models/{name}_epoch_{epoch+1}.pth')
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"Early stopping triggered. No improvement in validation loss for {patience} epochs.")
                break

    # After training, evaluate on test set if needed
    model.eval()
    test_running_loss = 0.0
    with torch.no_grad():
        for test_images, test_masks in test_dataloader:
            test_images = test_images.to(device)
            test_masks = test_masks.to(device)

            test_outputs = model(test_images)
            test_loss = criterion(test_outputs, test_masks)
            test_running_loss += test_loss.item()

    avg_test_loss = test_running_loss / len(test_dataloader)
    #print(f"Final Test Loss: {avg_test_loss:.4f}")

    #print("Training finished.")
    
    return model, train_losses, val_losses, avg_test_loss

def plot_training(name, train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color='deepskyblue')  # Nice blue color for training loss
    plt.plot(val_losses, label='Validation Loss', color='mediumorchid') # Pretty purple color for validation loss
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"{name}", fontsize=14, fontweight='bold')  # Bold title
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/learning_curves/{name}.png")
    #plt.show()


def prepare_data(dataset, batch_size=32):
    # Define the split ratios
    train_split = 0.8
    val_split = 0.1
    test_split = 0.1

    # Calculate lengths of each split
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    train_size = int(np.floor(train_split * dataset_size))
    val_size = int(np.floor(val_split * dataset_size))
    test_size = dataset_size - train_size - val_size

    # Shuffle and split indices
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Define samplers for obtaining batches
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Create data loaders
    train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=4)

    return train_dataloader, val_dataloader, test_dataloader

def run_train(dataset, loss = "xe", arch = "unet", balance = False, batch_size=32, seed=201, num_epochs = 30, gpu_index = 1):

    set_random_seed(seed=seed)

    train_dataloader, val_dataloader, test_dataloader = prepare_data(dataset, batch_size)

    # Calculate class weights
    class_weights = None
    if balance:
        class_weights = calculate_class_weights(dataset)

    if loss not in LossTypes.keys():
        raise ValueError("Invalid loss type. Loss type must be in: ", LossTypes.keys())
    if arch not in ArchTypes:
        ("Invalid loss type. Loss type must be in: ", ArchTypes)

    n_classes = LossTypes[loss]

    model = None
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

    name = f"labelbox_data_{arch}_{loss}_{'balanced' if balance else 'unbalanced'}_bs_{batch_size}_seed_{seed}"
    saved_name = f"models/{name}.pth"

    trained_model, train_losses, val_losses, avg_test_loss = train_model(model, loss, train_dataloader, val_dataloader, test_dataloader, name=name, class_weights=class_weights, num_epochs = num_epochs, gpu_num = gpu_index)

    torch.save(trained_model.state_dict(), saved_name)

    plot_training(name, train_losses, val_losses)

    #print(f"{name} training and plotting completed")

    return saved_name, avg_test_loss, name

if __name__ == "__main__":

    #run_name = "results/learning_curves/segnet_xe_transformed_seed_201.png"

    loss = "dice"

    arch = "DeepLabV3" #unet or nested unet or deeplabv3 or segnet

    balance = True

    dataset = CocoDataset(img_dir="training_images", ann_file="labelbox_coco.json", transform=transform)
    #dataset = CocoDataset(img_dir="training_labelbox/training_images", ann_file="training_labelbox/labelbox_coco.json", transform=transform)


    run_train(dataset, loss, arch=arch, balance=balance, num_epochs = 200, batch_size=64)