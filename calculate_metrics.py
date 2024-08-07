import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from unet import UNet
from archs.nested_unet import NestedUNet
from archs.deeplabv3 import DeepLabV3
from archs.segnet import SegNet
from coco_dataset import CocoDataset, transform
import random
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from train import calculate_class_weights
from postproccessing.confusion_matrix_evaluation import plot_confusion_matrix

# Set random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
# Ensure deterministic behavior for certain operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def compute_confusion_matrix(preds, labels):
    """Compute confusion matrix for binary segmentation."""
    preds_flat = preds.view(-1).cpu().numpy()
    labels_flat = labels.view(-1).cpu().numpy()

    # Compute confusion matrix
    cm = confusion_matrix(labels_flat, preds_flat, labels=[0, 1])

    return cm

def calculate_prf(conf_matrix):
    """Calculate precision, recall, and F1 score from confusion matrix."""
    TN, FP, FN, TP = conf_matrix.ravel()

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

    return precision, recall, f1

def weighted_mean(values, weights):
    if len(values) != len(weights):
        raise ValueError("The number of values and weights must be the same")
    
    # Calculate the weighted sum
    weighted_sum = sum(v * w for v, w in zip(values, weights))
    
    # Calculate the sum of the weights
    total_weight = sum(weights)
    
    # Calculate the weighted mean
    return weighted_sum / total_weight

def calculate_iou(pred, target, n_classes, dataset):
    ious = []
    smooth = 1e-6  # Smooth to avoid division by zero
    ious = []

    for cls in range(n_classes):
        pred_cls = (pred == cls).float().view(-1)
        target_cls = (target == cls).float().view(-1)


        intersection = (pred_cls * target_cls).sum().item()
        union = pred_cls.sum().item() + target_cls.sum().item() - intersection

        if union == 0:
            ious.append(1.0)  # If no ground truth or predictions for this class, IoU is perfect
        else:
            ious.append((intersection + smooth) / (union + smooth))

    if n_classes == 2:
        weights = calculate_class_weights(dataset)
        return np.mean(ious), weighted_mean(ious, weights.tolist())
    return np.mean(ious), np.mean(ious)


def calculate_dice(pred, target, smooth=1):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()

def evaluate_model(model, dataloader, device, n_classes, dataset):
    model.eval()
    running_iou = 0.0
    running_dice = 0.0
    running_iou_weighted = 0.0
    total_conf_matrix = np.zeros((2, 2))
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

            iou, iou_weighted = calculate_iou(preds, masks, n_classes, dataset)
            dice = calculate_dice(preds, masks)

            batch_conf_matrix = compute_confusion_matrix(preds, masks)
            total_conf_matrix += batch_conf_matrix

            running_iou += iou
            running_dice += dice
            running_iou_weighted += iou_weighted
            count += 1

    avg_iou = running_iou / count if count > 0 else float('nan')
    avg_iou_weighted = running_iou_weighted / count if count > 0 else float('nan')
    avg_dice = running_dice / count if count > 0 else float('nan')
    
    return avg_iou, avg_iou_weighted, avg_dice, total_conf_matrix

def run_metrics(trained_model, dataset, arch, batch, loss, subset_size = 200, gpu_index = 2):

    n_classes = 2
    if 'dice' == loss:
        n_classes = 1

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

    # Load the saved model weights
    model.load_state_dict(torch.load(trained_model))
    model.eval()

    # Select a subset of the dataset for testing
    subset_indices = random.sample(range(len(dataset)), subset_size)
    test_dataset = Subset(dataset, subset_indices)

    # Create the test data loader
    test_dataloader = DataLoader(test_dataset, batch_size=batch, num_workers=4)

    # Evaluate the model on the test set
    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    avg_iou, avg_iou_weighted, avg_test_dice, total_conf_matrix = evaluate_model(model, test_dataloader, device, n_classes, dataset)

    # print(f"Final Test IOU Weighted: {avg_iou:.4f}")
    # print(f"Final Test IOU Normal: {avg_iou_weighted:.4f}")
    # print(f"Final Test Dice Coefficient: {avg_test_dice:.4f}")

    precision, recall, f1 = calculate_prf(total_conf_matrix)
    return avg_iou, avg_iou_weighted, avg_test_dice, precision, recall, f1, total_conf_matrix

def main():
    trained_model = 'models/labelbox_data_DeepLabV3_dice_balanced_bs_64_seed_201_epoch_25.pth'
    dataset = CocoDataset(img_dir="training_images", ann_file="annotations/labelbox_coco.json", transform=transform)
    
    avg_iou, avg_iou_weighted, avg_test_dice, precision, recall, f1, total_conf_matrix = run_metrics(trained_model, dataset, arch = "DeepLabV3", batch = 32, loss="dice", subset_size = 500, gpu_index=2)
    
    # Plot cm and save to png to see output
    cm = plot_confusion_matrix(total_conf_matrix, classes=['Background', 'Leaf Hair'])

    output = {'Average IOU': float(avg_iou),
            'Average Dice Coefficient': avg_test_dice,
            'Precision': precision,
            'Recall': recall
            }
    print(output)
    

if __name__ == "__main__":
    main()
