import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torchvision import transforms, models
from torchvision.models.resnet import BasicBlock
from PIL import Image
from data_prep import Leaf_Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# Define paths and parameters
root_dir = '/Volumes/Image Data '
annotation_file = '/Volumes/Image Data /combined_coco.json'
batch_size = 32
shuffle_dataset = True
num_epochs = 10
num_classes = 2  # binary segmentation, set to 2 (background + object)

lr = 0.001

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5380782065015497, 0.6146645541178255, 0.4624397479931463], std=[0.12672495205043693, 0.12178723849002748, 0.1999076104405415])
])

# Instantiate your custom dataset
dataset = Leaf_Dataset(root_dir, annotation_file, transform=transform)

# Split dataset into training and validation sets
dataset_size = len(dataset)
val_split = 0.2  # 20% validation, adjust as needed
val_size = int(val_split * dataset_size)
train_size = dataset_size - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoader for training and validation
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_dataset)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.encoder = models.resnet34(pretrained=False)
        self.encoder_layers = list(self.encoder.children())
        
        self.conv1 = nn.Sequential(*self.encoder_layers[:3])
        self.pool1 = self.encoder_layers[3]
        self.conv2 = self.encoder_layers[4]
        self.conv3 = self.encoder_layers[5]
        self.conv4 = self.encoder_layers[6]
        self.conv5 = self.encoder_layers[7]

        self.upconv5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.decoder5 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.decoder4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.decoder3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.decoder2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.decoder1 = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x2 = self.pool1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x5 = self.conv4(x4)
        x6 = self.conv5(x5)

        # Decoder
        d5 = self.upconv5(x6)
        d5 = torch.cat((d5, x5), dim=1)
        d5 = torch.relu(self.decoder5(d5))
        
        d4 = self.upconv4(d5)
        d4 = torch.cat((d4, x4), dim=1)
        d4 = torch.relu(self.decoder4(d4))
        
        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, x3), dim=1)
        d3 = torch.relu(self.decoder3(d3))
        
        d2 = self.upconv2(d3)
        print(d3.shape)
        print(d2.shape)
        print(x2.shape)
        d2 = torch.cat((d2, x2), dim=1)
        print(d2.shape)
        print(x2.shape)
        d2 = torch.relu(self.decoder2(d2))
        
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, x1), dim=1)
        d1 = torch.relu(self.decoder1(d1))
        
        return d1

# Initialize your model
model = UNet(num_classes)

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)


# Create a summary writer
writer = SummaryWriter()
train_losses = []
val_losses = []
# Training loop
for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    for images, annotations in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, annotations)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
    
    # Compute average training loss for the epoch
    train_loss = running_train_loss / len(train_loader)
    
    # Validation loop
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images_val, annotations_val in val_loader:
            outputs_val = model(images_val)
            loss_val = criterion(outputs_val, annotations_val)
            running_val_loss += loss_val.item()
    
    # Compute average validation loss for the epoch
    val_loss = running_val_loss / len(val_loader)
    
    # Print or log the losses for each epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss}, Val Loss: {val_loss}")

writer.close()

# Inference example
model.eval()
with torch.no_grad():
    image_path = "/path/to/your/test/image.png"
    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0)
    outputs = model(input_tensor)
    predicted_masks = torch.argmax(outputs, dim=1)
    # Post-processing and visualization of segmentation results

import matplotlib.pyplot as plt

# Assuming you log your metrics during training
epochs = range(1, num_epochs + 1)
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()
