import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from unet import UNet  # Assuming you have a UNet model defined in unet.py
from coco_dataset import CocoDataset, transform  # Assuming you have a CocoDataset defined in coco_dataset.py
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

def confusion_matrix(preds, labels, num_classes):
    """Calculate the confusion matrix for a segmentation task."""
    preds = preds.view(-1).cpu().numpy()
    labels = labels.view(-1).cpu().numpy()

    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        conf_matrix[t, p] += 1
    return conf_matrix

def calculate_metrics(conf_matrix):
    """Calculate precision, recall, and F1 score from the confusion matrix."""
    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

def evaluate_model(model, dataloader, device, num_classes):
    """Evaluate the model and calculate the confusion matrix and metrics."""
    model.eval()
    total_conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            if num_classes == 1:
                preds = (outputs > 0.5).float()
            elif num_classes == 2:
                preds = outputs.argmax(dim=1)
            else:
                print("Wrong number of classes")

            total_conf_matrix += confusion_matrix(preds, masks, num_classes)

    precision, recall, f1_score = calculate_metrics(total_conf_matrix)

    return total_conf_matrix, precision, recall, f1_score

def main():
    model_path = 'models/unet_model_epoch_27.pth'
    isXE = False
    if 'xe' in model_path:
        isXE = True
    num_classes = 1  # Number of classes in your segmentation task
    if isXE:
        num_classes = 2

    model = UNet(3, num_classes)

    # Load the saved model weights
    model.load_state_dict(torch.load(model_path))
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
    conf_matrix, precision, recall, f1_score = evaluate_model(model, test_dataloader, device, num_classes)

    print("Confusion Matrix:\n", conf_matrix)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

if __name__ == "__main__":
    main()
