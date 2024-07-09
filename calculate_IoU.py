import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from unet import UNet
from coco_dataset import CocoDataset, transform
import random

def calculate_iou(pred, target, n_classes):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(n_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        
        if union == 0:
            ious.append(1.0)  # If there are no predictions or ground truth for this class, it's perfect (IoU=1)
        else:
            ious.append(intersection / union)
    
    # Filter out NaN values from the IOU calculation
    ious = [iou for iou in ious if not np.isnan(iou)]
    
    if len(ious) == 0:
        return float('nan')
    else:
        return np.mean(ious)

def evaluate_model(model, dataloader, device, n_classes):
    model.eval()
    running_iou = 0.0
    count = 0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            iou = calculate_iou(preds, masks, n_classes)
            if not np.isnan(iou):
                running_iou += iou
                count += 1

    avg_iou = running_iou / count if count > 0 else float('nan')
    return avg_iou

def main():
    # Define the model
    isXE = False
    n_classes = 1  # Number of classes in your segmentation task
    if isXE:
        n_classes = 2
    model = UNet(3, n_classes)

    # Load the saved model weights
    
    model.load_state_dict(torch.load('models/dice_loss_model.pth'))
    model.eval()

    # Load the full dataset
    full_dataset = CocoDataset(img_dir="Data", ann_file="Data/combined_coco.json", transform=transform)

    # Select a subset of the dataset for testing
    subset_size = 200  # Define the size of the subset
    subset_indices = random.sample(range(len(full_dataset)), subset_size)
    test_dataset = Subset(full_dataset, subset_indices)

    # Create the test data loader
    test_dataloader = DataLoader(test_dataset, batch_size=8, num_workers=4)

    # Evaluate the model on the test set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    avg_test_iou = evaluate_model(model, test_dataloader, device, n_classes)

    print(f"Final Test mIOU: {avg_test_iou:.4f}")

if __name__ == "__main__":
    main()
