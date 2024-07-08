import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from unet_model import UNet
from coco_dataset import CocoDataset, transform
import random
from get_class_frequencies import get_freqs

def calculate_iou(pred, target, n_classes, class_weights=None):
    ious = []
    pred = pred.view(-1).cpu()  # Move pred to CPU before operations
    target = target.view(-1).cpu()  # Move target to CPU before operations
    
    if class_weights is None:
        class_weights = torch.ones(n_classes).to(pred.device)
    else:
        class_weights = class_weights.to(pred.device)
    
    for cls in range(n_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        
        if union == 0:
            ious.append(1.0)  # If there are no predictions or ground truth for this class, it's perfect (IoU=1)
        else:
            iou = intersection / union
            weighted_iou = iou * class_weights[cls]
            ious.append(weighted_iou)
    
    # Filter out NaN values from the IOU calculation
    ious = [iou for iou in ious if not np.isnan(iou)]
    
    if len(ious) == 0:
        return float('nan')
    else:
        return np.sum(ious)

def evaluate_model(model, dataloader, device, n_classes, class_weights=None):
    model.eval()
    running_iou = 0.0
    count = 0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            iou = calculate_iou(preds, masks, n_classes, class_weights)
            if not np.isnan(iou):
                running_iou += iou
                count += 1

    avg_iou = running_iou / count if count > 0 else float('nan')
    return avg_iou

def main():
    # Define the model
    n_classes = 2  # Adjust based on your number of classes
    model = UNet(n_classes)

    # Load the saved model weights
    model.load_state_dict(torch.load('models/balanced_by_inverse_unet_model_epoch_25.pth'))
    model.eval()

    # Load the full dataset
    full_dataset = CocoDataset(img_dir="Data", ann_file="Data/combined_coco.json", transform=transform)

    # Select a subset of the dataset for testing
    subset_size = 200  # Define the size of the subset
    subset_indices = random.sample(range(len(full_dataset)), subset_size)
    test_dataset = Subset(full_dataset, subset_indices)

    # Create the test data loader
    test_dataloader = DataLoader(test_dataset, batch_size=8, num_workers=4)

    # Define class weights (example)
    # Get class weights
    annotations = "Data/combined_coco.json"
    full_dataset = CocoDataset(img_dir="Data", ann_file=annotations, transform=transform)

    class_weights = torch.tensor(get_freqs(annotations, True), dtype=torch.float32)

    # Evaluate the model on the test set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    avg_test_iou = evaluate_model(model, test_dataloader, device, n_classes, class_weights)

    print(f"Final Test mIOU: {avg_test_iou:.4f}")

if __name__ == "__main__":
    main()
