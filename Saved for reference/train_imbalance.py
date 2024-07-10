#to be deleted, for reference currently

# import torch
# import torch.optim as optim
# import torch.nn as nn
# from unet_model import UNet
# from coco_dataset import CocoDataset, transform
# from torch.utils.data import DataLoader, SubsetRandomSampler
# import time
# import numpy as np
# import matplotlib.pyplot as plt
# from get_class_frequencies import get_freqs

# def train_model():
#     # Define the model, loss function, and optimizer
#     n_classes = 2  # Adjust based on your number of classes
#     model = UNet(n_classes)
    
#     # Load full dataset
#     annotations = "Data/combined_coco.json"
#     full_dataset = CocoDataset(img_dir="Data", ann_file=annotations, transform=transform)

#     # Define device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)


#     # Get class weights
#     class_weights = torch.tensor(get_freqs(annotations, True), dtype=torch.float32)

#     # Define criterion with class weights
#     criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

#     # Define optimizer
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     # Define data loaders
#     train_split = 0.8
#     val_split = 0.1
#     test_split = 0.1

#     dataset_size = len(full_dataset)
#     indices = list(range(dataset_size))
#     np.random.shuffle(indices)
#     train_size = int(np.floor(train_split * dataset_size))
#     val_size = int(np.floor(val_split * dataset_size))
#     test_size = dataset_size - train_size - val_size

#     train_indices = indices[:train_size]
#     val_indices = indices[train_size:train_size + val_size]
#     test_indices = indices[train_size + val_size:]

#     train_sampler = SubsetRandomSampler(train_indices)
#     val_sampler = SubsetRandomSampler(val_indices)
#     test_sampler = SubsetRandomSampler(test_indices)

#     train_loader = DataLoader(full_dataset, batch_size=32, sampler=train_sampler, num_workers=4)
#     val_loader = DataLoader(full_dataset, batch_size=32, sampler=val_sampler, num_workers=4)
#     test_loader = DataLoader(full_dataset, batch_size=32, sampler=test_sampler, num_workers=4)

#     # Training and validation loop
#     num_epochs = 30
#     train_losses = []
#     val_losses = []

#     for epoch in range(num_epochs):
#         # Training phase
#         model.train()
#         running_loss = 0.0
#         start_time = time.time()
#         for images, masks in train_loader:
#             images = images.to(device)
#             masks = masks.to(device)

#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, masks)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#         epoch_time = time.time() - start_time
#         avg_train_loss = running_loss / len(train_loader)
#         train_losses.append(avg_train_loss)
#         print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Time: {epoch_time:.2f} seconds")

#         # Validation phase
#         model.eval()
#         val_running_loss = 0.0
#         with torch.no_grad():
#             for val_images, val_masks in val_loader:
#                 val_images = val_images.to(device)
#                 val_masks = val_masks.to(device)

#                 val_outputs = model(val_images)
#                 val_loss = criterion(val_outputs, val_masks)
#                 val_running_loss += val_loss.item()

#         avg_val_loss = val_running_loss / len(val_loader)
#         val_losses.append(avg_val_loss)
#         print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

#         # Save model checkpoint
#         torch.save(model.state_dict(), f'models/unet_model_epoch_{epoch+1}.pth')

#     # After training, evaluate on test set if needed
#     model.eval()
#     test_running_loss = 0.0
#     with torch.no_grad():
#         for test_images, test_masks in test_loader:
#             test_images = test_images.to(device)
#             test_masks = test_masks.to(device)

#             test_outputs = model(test_images)
#             test_loss = criterion(test_outputs, test_masks)
#             test_running_loss += test_loss.item()

#     avg_test_loss = test_running_loss / len(test_loader)
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
#     train_model()