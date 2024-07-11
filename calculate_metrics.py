import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from unet import UNet
from coco_dataset import CocoDataset, transform
import random

# Set random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
# Ensure deterministic behavior for certain operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def calculate_iou(pred, target, n_classes):
    # ious = []
    # pred = pred.view(-1)
    # target = target.view(-1)
    # cls = 1 if n_classes == 2 else 0

    # pred_inds = (pred == cls)
    # target_inds = (target == cls)
    
    # intersection = (pred_inds[target_inds]).long().sum().item()
    # union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
    
    # if union == 0:
    #     ious.append(1.0)  # If there are no predictions or ground truth for this class, it's perfect (IoU=1)
    # else:
    #     ious.append(intersection / union)
    
    # # Filter out NaN values from the IOU calculation
    # ious = [iou for iou in ious if not np.isnan(iou)]
    # if len(ious) == 0:
    #     return float('nan')
    # else:
    #     return np.mean(ious)
    

    smooth = 1e-6  # Smooth to avoid division by zero
    ious = []
    tp_ious = []

    for cls in range(n_classes):
        pred_cls = (pred == cls).float().view(-1)
        target_cls = (target == cls).float().view(-1)

        intersection = (pred_cls * target_cls).sum().item()
        union = pred_cls.sum().item() + target_cls.sum().item() - intersection

        tp_intersection = (pred_cls & target_cls).float().sum().item()
        tp_union = (pred_cls | target_cls).float().sum().item()

        if union == 0:
            ious.append(1.0)  # If no ground truth or predictions for this class, IoU is perfect
        else:
            ious.append((intersection + smooth) / (union + smooth))

        if union != 0:
            tp_ious.append((tp_intersection + smooth) / (tp_union + smooth))
    return ious, tp_ious

def calculate_dice(pred, target, smooth=1):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()

def evaluate_model(model, dataloader, device, n_classes):
    model.eval()
    running_iou = 0.0
    running_dice = 0.0
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

            iou = calculate_iou(preds, masks, n_classes)
            dice = calculate_dice(preds, masks)
            
            if not np.isnan(iou):
                running_iou += iou
                running_dice += dice
                count += 1

    avg_iou = running_iou / count if count > 0 else float('nan')
    avg_dice = running_dice / count if count > 0 else float('nan')
    return avg_iou, avg_dice

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
    avg_test_iou, avg_test_dice = evaluate_model(model, test_dataloader, device, n_classes)

    print(f"Final Test mIOU: {avg_test_iou:.4f}")
    print(f"Final Test Dice Coefficient: {avg_test_dice:.4f}")

if __name__ == "__main__":
    main()
