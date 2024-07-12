#for reference

# import torch
# import numpy as np
# from sklearn.metrics import confusion_matrix
# from unet import UNet
# from coco_dataset import CocoDataset, transform
# from torch.utils.data import DataLoader, SubsetRandomSampler

# def compute_iou(y_pred, y_true):
#     """
#     Compute Intersection over Union (IoU) for semantic segmentation.

#     Args:
#         y_pred (torch.Tensor): Predicted labels (batch_size, height, width).
#         y_true (torch.Tensor): True labels (batch_size, height, width).

#     Returns:
#         float: Mean IoU across batch samples.
#     """
#     eps = 1e-6
#     intersection = torch.logical_and(y_pred, y_true).sum((1, 2)).float()
#     union = torch.logical_or(y_pred, y_true).sum((1, 2)).float()
#     iou = (intersection + eps) / (union + eps)
#     return iou.mean().item()

# def evaluate_model(model, data_loader):
#     model.eval()
#     total_iou = 0.0
#     with torch.no_grad():
#         for images, masks in data_loader:
#             images = images.to(device)
#             masks = masks.to(device)

#             outputs = model(images)
#             predictions = (outputs > 0.5).float()  # Adjust threshold as per your need

#             # Calculate IoU for the batch
#             batch_iou = compute_iou(predictions, masks)
#             total_iou += batch_iou

#     mean_iou = total_iou / len(data_loader)
#     return mean_iou

# if __name__ == "__main__":
#     # Assuming 'trained_model' is your trained UNet model
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = UNet.load_model("models/unet_model_epoch_29.pth", 3, 1).to(device)
#     full_dataset = CocoDataset(img_dir="Data", ann_file="Data/combined_coco.json", transform=transform)
#     iou_dataloader = DataLoader(full_dataset, batch_size=32, num_workers=4)
#     # Assuming you have separate data loaders for validation and test sets
#     iou = evaluate_model(model, iou_dataloader)

#     print(f"Validation mIoU: {iou:.4f}")