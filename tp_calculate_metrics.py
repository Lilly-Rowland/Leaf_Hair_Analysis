import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from unet import UNet
from coco_dataset import CocoDataset, transform
import random
true_positives_total = 0
# Set random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
# Ensure deterministic behavior for certain operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def calculate_iou_tp(pred, target, n_classes):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    cls = 1 if n_classes == 2 else 0
    pred_inds = (pred == cls)
    target_inds = (target == cls)
    intersection = (pred_inds & target_inds).long().sum().item()
    union = (pred_inds | target_inds).long().sum().item()
    
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

def calculate_true_positives(pred, target, n_classes):

    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    cls = 1 if n_classes == 2 else 0
    
    pred_inds = (pred_flat == cls)
    target_inds = (target_flat == cls)
    
    true_positives = (pred_inds & target_inds).sum().item()
    global true_positives_total
    true_positives_total += true_positives
    return true_positives

def calculate_dice(pred, target, smooth=1):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()

def evaluate_model(model, dataloader, device, n_classes):
    model.eval()
    running_iou_tp = 0.0
    running_dice = 0.0
    running_tp_leaf_hair = 0
    count = 0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)

            if n_classes == 1:
                preds = (outputs > 0.5).float()
            elif n_classes == 2:
                preds = outputs.argmax(dim=1)
            else:
                print("Wrong number of classes")

            iou_tp = calculate_iou_tp(preds, masks, n_classes)
            dice = calculate_dice(preds, masks)
            tp_leaf_hair = calculate_true_positives(preds, masks, n_classes)
            
            if not np.isnan(iou_tp):
                running_iou_tp += iou_tp
                running_dice += dice
                running_tp_leaf_hair += tp_leaf_hair
                count += 1

    avg_iou_tp = running_iou_tp / count if count > 0 else float('nan')
    avg_dice = running_dice / count if count > 0 else float('nan')
    avg_tp_leaf_hair = running_tp_leaf_hair / count if count > 0 else float('nan')
    
    return avg_iou_tp, avg_dice, avg_tp_leaf_hair

def main():
    
    name = 'models/unet_model_epoch_1.pth'
    isXE = False
    if 'xe' in name:
        isXE = True
    n_classes = 1  # Number of classes in your segmentation task
    if isXE:
        n_classes = 2
    model = UNet(3, n_classes)

    # Load the saved model weights
    model.load_state_dict(torch.load(name))
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
    avg_test_iou, avg_test_dice, _ = evaluate_model(model, test_dataloader, device, n_classes)

    print(f"Final Test mIOU: {avg_test_iou:.4f}")
    print(f"Final Test Dice Coefficient: {avg_test_dice:.4f}")
    print(true_positives_total)

if __name__ == "__main__":
    main()
