import torch
import torch.optim as optim
import torch.nn as nn
from unet import UNet  # Import your UNet model class
from coco_dataset import CocoDataset, transform  # Import your dataset class and transformation function
from torch.utils.data import DataLoader, SubsetRandomSampler
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dice_loss import DiceLoss, WeightedDiceLoss
from enum import Enum
import random

# Set random seed for reproducibility
random_seed = 201
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.cuda.empty_cache()

class Criterion(Enum):
    DICE = 1
    XE = 2

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
    print(f"{weight_background} and {weight_foreground}")

    return torch.tensor([weight_background, weight_foreground])


def train_model(model, loss, train_loader, val_loader, test_dataloader, class_weights = None, lr=0.001):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if class_weights == None:
        class_weights = torch.ones(2, dtype=torch.float32)
    class_weights = class_weights.to(device)

    if loss.upper() == "DICE":
        #if class_weights is None:
            #class_weights = torch.ones(1, dtype=torch.float32)
        criterion = WeightedDiceLoss(weight=class_weights)
    elif loss.upper() == "XE":
        #if class_weights is None:
            #class_weights = torch.ones(2, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        print("Invalid criterion")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    
    num_epochs = 30
    train_losses = []
    val_losses = []

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
        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

        # Save model checkpoint
        torch.save(model.state_dict(), f'models/unet_model_epoch_{epoch+1}.pth')


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
    print(f"Final Test Loss: {avg_test_loss:.4f}")

    print("Training finished.")
    
    return model, train_losses, val_losses

def plot_training(name, train_losses, val_losses):
# Plotting the learning curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(name)
    plt.show()

def prepare_data(dataset):
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
    train_dataloader = DataLoader(dataset, batch_size=32, sampler=train_sampler, num_workers=4)
    val_dataloader = DataLoader(dataset, batch_size=32, sampler=val_sampler, num_workers=4)
    test_dataloader = DataLoader(dataset, batch_size=32, sampler=test_sampler, num_workers=4)

    return train_dataloader, val_dataloader, test_dataloader

def run_train(dataset, run_name = "learning_curve.png", loss = "XE", balance = False):

    dataset = CocoDataset(img_dir="Data", ann_file="Data/combined_coco.json", transform=transform)

    train_dataloader, val_dataloader, test_dataloader = prepare_data(dataset)

    class_weights = None
    if balance:
    # Calculate class weights
        class_weights = calculate_class_weights(dataset)

    n_classes = 1

    if loss == "XE":
        n_classes = 2

    model = UNet(3, n_classes)  # Example: Replace with your UNet model instantiation

    trained_model, train_losses, val_losses = train_model(model, loss, train_dataloader, val_dataloader, test_dataloader, class_weights=class_weights)
    
    plot_training(run_name, train_losses, val_losses)

if __name__ == "__main__":

    run_name = "results/learning_curves/weighted_dice_transformed_seed_201.png"

    loss = "Dice"

    balance = True

    dataset = CocoDataset(img_dir="Data", ann_file="Data/combined_coco.json", transform=transform)


    run_train(dataset, run_name, loss, balance=balance)