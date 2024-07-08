# import torch
# import torch.optim as optim
# import torch.nn as nn
# from unet_model import UNet
# from coco_dataset_dice import CocoDataset, transform
# from torch.utils.data import DataLoader, SubsetRandomSampler
# import time
# import numpy as np
# import matplotlib.pyplot as plt
# from get_class_frequencies import get_freqs
# import torch.nn.functional as F


# class DiceBCELoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceBCELoss, self).__init__()

#     def forward(self, inputs, targets, smooth=1):
        
#         #comment out if your model contains a sigmoid or equivalent activation layer
#         #inputs = F.sigmoid(inputs)       

#         inputs = torch.as_tensor(inputs, dtype=torch.float32)
#         targets = torch.as_tensor(targets, dtype=torch.float32)
#         print(f"Tiput Image Size: {inputs.size()}, Trarget Mask Size: {targets.size()}")

#         inputs.squeeze()
#         targets.squeeze()
#         print(f"Tiput Image Size: {inputs.size()}, Trarget Mask Size: {targets.size()}")

        
#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
        
#         intersection = (inputs * targets).sum()                            
#         dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
#         BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
#         Dice_BCE = BCE + dice_loss
        
#         return Dice_BCE

# def train_model(batch_size=32, lr=0.001):
#     # Define the model, loss function, and optimizer
#     num_classes = 10  # Replace with the actual number of classes
#     model = UNet(num_classes)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     annotations = "Data/combined_coco.json"
#     full_dataset = CocoDataset(img_dir="Data", ann_file=annotations, transform=transform, num_classes=num_classes)
    
#     weighted = False
#     # Get class weights
#     if weighted:
#         class_weights = torch.tensor(get_freqs(annotations, True), dtype=torch.float32)
#         criterion = DiceBCELoss(class_weights=class_weights.to(device))
#     else:
#         criterion = DiceBCELoss()

#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     # Define the split ratios
#     train_split = 0.8
#     val_split = 0.1
#     test_split = 0.1

#     # Calculate lengths of each split
#     dataset_size = len(full_dataset)
#     indices = list(range(dataset_size))
#     train_size = int(np.floor(train_split * dataset_size))
#     val_size = int(np.floor(val_split * dataset_size))
#     test_size = dataset_size - train_size - val_size

#     # Shuffle and split indices
#     np.random.shuffle(indices)
#     train_indices = indices[:train_size]
#     val_indices = indices[train_size:train_size + val_size]
#     test_indices = indices[train_size + val_size:]

#     # Define samplers for obtaining batches
#     train_sampler = SubsetRandomSampler(train_indices)
#     val_sampler = SubsetRandomSampler(val_indices)
#     test_sampler = SubsetRandomSampler(test_indices)

#     # Create data loaders
#     train_dataloader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
#     val_dataloader = DataLoader(full_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4)
#     test_dataloader = DataLoader(full_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=4)

#     # Training and validation loop
#     num_epochs = 30
#     model.to(device)

#     train_losses = []
#     val_losses = []

#     for epoch in range(num_epochs):
#         # Training phase
#         model.train()
#         running_loss = 0.0
#         start_time = time.time()
#         for images, masks in train_dataloader:
#             images = images.to(device)
#             masks = masks.to(device)

#             optimizer.zero_grad()
#             outputs = model(images)
            
#             # Assuming outputs from model have dimensions [batch_size, num_classes, height, width]
#             # Convert integer masks to one-hot format
#             masks_one_hot = F.one_hot(masks, num_classes=num_classes).permute(0, 3, 1, 2).float()
            
#             loss = criterion(outputs, masks_one_hot)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#         epoch_time = time.time() - start_time
#         avg_train_loss = running_loss / len(train_dataloader)
#         train_losses.append(avg_train_loss)
#         print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Time: {epoch_time:.2f} seconds")

#         # Validation phase
#         model.eval()  # Switch to evaluation mode
#         val_running_loss = 0.0
#         with torch.no_grad():
#             for val_images, val_masks in val_dataloader:
#                 val_images = val_images.to(device)
#                 val_masks = val_masks.to(device)

#                 val_outputs = model(val_images)
#                 val_loss = criterion(val_outputs, val_masks)
#                 val_running_loss += val_loss.item()

#         avg_val_loss = val_running_loss / len(val_dataloader)
#         val_losses.append(avg_val_loss)
#         print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

#         # Save model checkpoint
#         torch.save(model.state_dict(), f'models/unet_model_epoch_{epoch+1}.pth')

#     # After training, evaluate on test set if needed
#     model.eval()
#     test_running_loss = 0.0
#     with torch.no_grad():
#         for test_images, test_masks in test_dataloader:
#             test_images = test_images.to(device)
#             test_masks = test_masks.to(device)

#             test_outputs = model(test_images)
#             test_loss = criterion(test_outputs, test_masks)
#             test_running_loss += test_loss.item()

#     avg_test_loss = test_running_loss / len(test_dataloader)
#     print(f"Final Test Loss: {avg_test_loss:.4f}")

#     # Plotting the learning curves
#     plt.figure(figsize=(10, 5))
#     plt.plot(train_losses, label='Training Loss')
#     plt.plot(val_losses, label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Loss')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# if __name__ == "__main__":
#     train_model(32, 0.001)


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

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

def train_model(model, train_loader, val_loader, test_dataloader, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = DiceLoss()
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
    plt.savefig("")
    plt.show(name)

if __name__ == "__main__":
    # Example usage
    full_dataset = CocoDataset(img_dir="Data", ann_file="Data/combined_coco.json", transform=transform)

    # Define the split ratios
    train_split = 0.8
    val_split = 0.1
    test_split = 0.1

    # Calculate lengths of each split
    dataset_size = len(full_dataset)
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
    train_dataloader = DataLoader(full_dataset, batch_size=32, sampler=train_sampler, num_workers=4)
    val_dataloader = DataLoader(full_dataset, batch_size=32, sampler=val_sampler, num_workers=4)
    test_dataloader = DataLoader(full_dataset, batch_size=32, sampler=test_sampler, num_workers=4)


    model = UNet(3,1)  # Example: Replace with your UNet model instantiation

    trained_model, train_losses, val_losses = train_model(model, train_dataloader, val_dataloader, test_dataloader)
    name = "dice_loss_32_batch.png"
    plot_training(name, train_losses, val_losses)