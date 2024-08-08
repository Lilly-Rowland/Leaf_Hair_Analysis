import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from unet import UNet  # Assuming your UNet model implementation is in unet.py
from coco_dataset import CocoDataset, transform  # Assuming your dataset loading code is in coco_dataset.py
import random

# Set random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def compute_confusion_matrix(preds, labels):
    """Compute confusion matrix for binary segmentation."""
    preds_flat = preds.view(-1).cpu().numpy()
    labels_flat = labels.view(-1).cpu().numpy()

    # Compute confusion matrix
    cm = confusion_matrix(labels_flat, preds_flat, labels=[0, 1])

    return cm

def plot_confusion_matrix(cm, classes, name = "conf_mat.png"):
    """Plot confusion matrix."""
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    print(name)
    plt.savefig(name)
    #plt.show()

def evaluate_model(model, dataloader, device):
    """Evaluate model and compute confusion matrix."""
    model.eval()
    total_conf_matrix = np.zeros((2, 2))
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()  # Adjust threshold as needed for your binary segmentation
            
            # Collect predictions and labels
            all_preds.append(preds.cpu().numpy())
            all_labels.append(masks.cpu().numpy())
            
            # Compute confusion matrix for this batch
            batch_conf_matrix = compute_confusion_matrix(preds, masks)
            total_conf_matrix += batch_conf_matrix
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return total_conf_matrix, all_preds, all_labels

def calculate_metrics(conf_matrix):
    """Calculate precision, recall, and F1 score from confusion matrix."""
    TN, FP, FN, TP = conf_matrix.ravel()

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

    return precision, recall, f1

def main():
    # Load your pretrained model
    model_path = 'models/unet_model_epoch_27.pth'
    n_classes = 1  # Binary segmentation task

    model = UNet(3, n_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load test dataset
    dataset = CocoDataset(img_dir="Data", ann_file="Data/combined_coco.json", transform=transform)
    subset_size = 200  # Subset size for faster evaluation
    subset_indices = random.sample(range(len(dataset)), subset_size)
    test_dataset = Subset(dataset, subset_indices)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Evaluate model and compute confusion matrix
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    conf_matrix, all_preds, all_labels = evaluate_model(model, test_dataloader, device)

    # Calculate metrics
    precision, recall, f1 = calculate_metrics(conf_matrix)

    # Print metrics
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # Define class labels (adjust based on your specific task)
    class_names = ['Background', 'Leaf Hair']

    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix, classes=class_names)

if __name__ == "__main__":
    main()
