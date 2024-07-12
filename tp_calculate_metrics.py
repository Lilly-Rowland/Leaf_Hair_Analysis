# # import torch
# # import numpy as np
# # from torch.utils.data import DataLoader, Subset
# # from unet import UNet
# from archs.nested_unet import NestedUNet
# from archs.deeplabv3 import DeepLabV3
# from archs.segnet import SegNet
# # from coco_dataset import CocoDataset, transform
# # import random
# # true_positives_total = 0
# # # Set random seed for reproducibility
# # random_seed = 42
# # random.seed(random_seed)
# # np.random.seed(random_seed)
# # torch.manual_seed(random_seed)
# # torch.cuda.manual_seed_all(random_seed)
# # # Ensure deterministic behavior for certain operations
# # torch.backends.cudnn.deterministic = True
# # torch.backends.cudnn.benchmark = False

# # def calculate_iou_tp(pred, target, n_classes):
# #     ious = []
# #     pred = pred.view(-1)
# #     target = target.view(-1)
# #     cls = 1 if n_classes == 2 else 0
# #     pred_inds = (pred == cls)
# #     target_inds = (target == cls)
# #     intersection = (pred_inds & target_inds).long().sum().item()
# #     union = (pred_inds | target_inds).long().sum().item()
    
# #     if union == 0:
# #         ious.append(1.0)  # If there are no predictions or ground truth for this class, it's perfect (IoU=1)
# #     else:
# #         ious.append(intersection / union)
    
# #     # Filter out NaN values from the IOU calculation
# #     ious = [iou for iou in ious if not np.isnan(iou)]
# #     if len(ious) == 0:
# #         return float('nan')
# #     else:
# #         return np.mean(ious)

# # def calculate_true_positives(pred, target, n_classes):

# #     pred_flat = pred.view(-1)
# #     target_flat = target.view(-1)
# #     cls = 1 if n_classes == 2 else 0
    
# #     pred_inds = (pred_flat == cls)
# #     target_inds = (target_flat == cls)
    
# #     true_positives = (pred_inds & target_inds).sum().item()
# #     global true_positives_total
# #     true_positives_total += true_positives
# #     return true_positives

# # def calculate_dice(pred, target, smooth=1):
# #     pred = pred.view(-1)
# #     target = target.view(-1)
# #     intersection = (pred * target).sum()
# #     dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
# #     return dice.item()

# # def evaluate_model(model, dataloader, device, n_classes):
# #     model.eval()
# #     running_iou_tp = 0.0
# #     running_dice = 0.0
# #     running_tp_leaf_hair = 0
# #     count = 0
    
# #     with torch.no_grad():
# #         for images, masks in dataloader:
# #             images = images.to(device)
# #             masks = masks.to(device)
            
# #             outputs = model(images)

# #             if n_classes == 1:
# #                 preds = (outputs > 0.5).float()
# #             elif n_classes == 2:
# #                 preds = outputs.argmax(dim=1)
# #             else:
# #                 print("Wrong number of classes")

# #             iou_tp = calculate_iou_tp(preds, masks, n_classes)
# #             dice = calculate_dice(preds, masks)
# #             tp_leaf_hair = calculate_true_positives(preds, masks, n_classes)
            
# #             if not np.isnan(iou_tp):
# #                 running_iou_tp += iou_tp
# #                 running_dice += dice
# #                 running_tp_leaf_hair += tp_leaf_hair
# #                 count += 1

# #     avg_iou_tp = running_iou_tp / count if count > 0 else float('nan')
# #     avg_dice = running_dice / count if count > 0 else float('nan')
    
# #     return avg_iou_tp, avg_dice

# # def run_metrics(trained_model, dataset, arch, batch):

# #     n_classes = 1
# #     if 'xe' == arch:
# #         n_classes = 2


# #     model = UNet(3, n_classes)

# #     # Load the saved model weights
# #     model.load_state_dict(torch.load(trained_model))
# #     model.eval()

# #     # Select a subset of the dataset for testing
# #     subset_size = 200  # Define the size of the subset
# #     subset_indices = random.sample(range(len(dataset)), subset_size)
# #     test_dataset = Subset(dataset, subset_indices)

# #     # Create the test data loader
# #     test_dataloader = DataLoader(test_dataset, batch_size=batch, num_workers=4)

# #     # Evaluate the model on the test set
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     model.to(device)
# #     avg_test_iou, avg_test_dice, avg_tp_ious = evaluate_model(model, test_dataloader, device, n_classes)

# #     print(f"Final Test mIOU: {avg_test_iou:.4f}")
# #     print(f"Final Test Dice Coefficient: {avg_test_dice:.4f}")
# #     print(f"Final Test TP mIOU: {avg_tp_ious:.4f}")


# # def main():
    
# #     name = 'models/unet_model_epoch_28.pth'
# #     arch = "segnet"
# #     isXE = True
# #     if 'xe' in name:
# #         isXE = True
# #     n_classes = 1  # Number of classes in your segmentation task
# #     if isXE:
# #         n_classes = 2

# #     model = None
# #     if arch.lower() == "unet":
# #         model = UNet(3, n_classes)  # Example: Replace with your UNet model instantiation
# #     elif arch.lower() == "nested_unet":
# #         model = NestedUNet(3, n_classes)
# #     elif arch.lower() == "deeplabv3":
# #         model = DeepLabV3(num_classes=n_classes)
# #     elif arch.lower() == "segnet":
# #         model = SegNet(3, n_classes)
# #     else:
# #         print("Invalid model")

# #     # Load the saved model weights
# #     model.load_state_dict(torch.load(name))
# #     model.eval()

# #     # Load the full dataset
# #     full_dataset = CocoDataset(img_dir="Data", ann_file="Data/combined_coco.json", transform=transform)

# #     # Select a subset of the dataset for testing
# #     subset_size = 200  # Define the size of the subset
# #     subset_indices = random.sample(range(len(full_dataset)), subset_size)
# #     test_dataset = Subset(full_dataset, subset_indices)

# #     # Create the test data loader
# #     test_dataloader = DataLoader(test_dataset, batch_size=8, num_workers=4)

# #     # Evaluate the model on the test set
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     model.to(device)
# #     avg_test_iou, avg_test_dice, avg_tp_ious = evaluate_model(model, test_dataloader, device, n_classes)

# #     print(f"Final Test mIOU: {avg_test_iou:.4f}")
# #     print(f"Final Test Dice Coefficient: {avg_test_dice:.4f}")
# #     print(f"Final Test TP mIOU: {avg_tp_ious:.4f}")

# # if __name__ == "__main__":
# #     main()


# import torch
# import numpy as np
# from torch.utils.data import DataLoader, Subset
# from unet import UNet
# from archs.nested_unet import NestedUNet
# from archs.deeplabv3 import DeepLabV3
# from archs.segnet import SegNet
# from coco_dataset import CocoDataset, transform
# import random

# # Set random seed for reproducibility
# random_seed = 42
# random.seed(random_seed)
# np.random.seed(random_seed)
# torch.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# def calculate_iou(pred, target, n_classes):
#     ious_tp = []
#     ious_normal = []
#     pred_flat = pred.view(-1)
#     target_flat = target.view(-1)
#     cls = 1 if n_classes == 2 else 0
    
#     pred_inds = (pred_flat == cls)
#     target_inds = (target_flat == cls)
    
#     intersection = (pred_inds & target_inds).long().sum().item()
#     union = (pred_inds | target_inds).long().sum().item()
    
#     if union == 0:
#         ious_tp.append(1.0)  # If there are no predictions or ground truth for this class, it's perfect (IoU=1)
#         ious_normal.append(1.0)
#     else:
#         ious_tp.append(intersection / union)
        
#         # Calculate normal IOU for all classes
#         intersection_all = (pred_flat & target_flat).long().sum().item()
#         union_all = (pred_flat | target_flat).long().sum().item()
#         ious_normal.append(intersection_all / union_all)
    
#     # Filter out NaN values from the IOU calculation
#     ious_tp = [iou for iou in ious_tp if not np.isnan(iou)]
#     ious_normal = [iou for iou in ious_normal if not np.isnan(iou)]
    
#     avg_iou_tp = np.mean(ious_tp) if len(ious_tp) > 0 else float('nan')
#     avg_iou_normal = np.mean(ious_normal) if len(ious_normal) > 0 else float('nan')
    
#     return avg_iou_tp, avg_iou_normal

# def calculate_true_positives(pred, target, n_classes):
#     pred_flat = pred.view(-1)
#     target_flat = target.view(-1)
#     cls = 1 if n_classes == 2 else 0
    
#     pred_inds = (pred_flat == cls)
#     target_inds = (target_flat == cls)
    
#     true_positives = (pred_inds & target_inds).sum().item()
    
#     return true_positives

# def calculate_dice(pred, target, smooth=1):
#     pred = pred.view(-1)
#     target = target.view(-1)
#     intersection = (pred * target).sum()
#     dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
#     return dice.item()

# def evaluate_model(model, dataloader, device, n_classes):
#     model.eval()
#     running_iou_tp = 0.0
#     running_iou_normal = 0.0
#     running_dice = 0.0
#     running_tp_leaf_hair = 0
#     count = 0
    
#     with torch.no_grad():
#         for images, masks in dataloader:
#             images = images.to(device)
#             masks = masks.to(device)
            
#             outputs = model(images)

#             if n_classes == 1:
#                 preds = (outputs > 0.5).float()
#             elif n_classes == 2:
#                 preds = outputs.argmax(dim=1)
#             else:
#                 print("Wrong number of classes")

#             iou_tp, iou_normal = calculate_iou(preds, masks, n_classes)
#             dice = calculate_dice(preds, masks)
#             tp_leaf_hair = calculate_true_positives(preds, masks, n_classes)
            
#             if not np.isnan(iou_tp):
#                 running_iou_tp += iou_tp
#                 running_iou_normal += iou_normal
#                 running_dice += dice
#                 running_tp_leaf_hair += tp_leaf_hair
#                 count += 1

#     avg_iou_tp = running_iou_tp / count if count > 0 else float('nan')
#     avg_iou_normal = running_iou_normal / count if count > 0 else float('nan')
#     avg_dice = running_dice / count if count > 0 else float('nan')
    
#     return avg_iou_tp, avg_iou_normal, avg_dice

# def run_metrics(trained_model, dataset, arch, batch):

#     n_classes = 1
#     if 'xe' == arch:
#         n_classes = 2

#     model = UNet(3, n_classes)

#     # Load the saved model weights
#     model.load_state_dict(torch.load(trained_model))
#     model.eval()

#     # Select a subset of the dataset for testing
#     subset_size = 200  # Define the size of the subset
#     subset_indices = random.sample(range(len(dataset)), subset_size)
#     test_dataset = Subset(dataset, subset_indices)

#     # Create the test data loader
#     test_dataloader = DataLoader(test_dataset, batch_size=batch, num_workers=4)

#     # Evaluate the model on the test set
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     avg_test_iou_tp, avg_test_iou_normal, avg_test_dice = evaluate_model(model, test_dataloader, device, n_classes)

#     print(f"Final Test IOU TP: {avg_test_iou_tp:.4f}")
#     print(f"Final Test IOU Normal: {avg_test_iou_normal:.4f}")
#     print(f"Final Test Dice Coefficient: {avg_test_dice:.4f}")

# def main():
    
#     trained_model_path = 'models/dice_loss_3-1_param.pth'
#     arch = "unet"
#     isXE = False
#     if 'xe' in trained_model_path:
#         isXE = True
#     n_classes = 1  # Number of classes in your segmentation task
#     if isXE:
#         n_classes = 2

    # model = None
    # if arch.lower() == "unet":
    #     model = UNet(3, n_classes)  # Example: Replace with your UNet model instantiation
    # elif arch.lower() == "nested_unet":
    #     model = NestedUNet(3, n_classes)
    # elif arch.lower() == "deeplabv3":
    #     model = DeepLabV3(num_classes=n_classes)
    # elif arch.lower() == "segnet":
    #     model = SegNet(3, n_classes)
    # else:
    #     print("Invalid model")

#     # Load the saved model weights
#     model.load_state_dict(torch.load(trained_model_path))
#     model.eval()

#     # Load the full dataset
#     full_dataset = CocoDataset(img_dir="Data", ann_file="Data/combined_coco.json", transform=transform)

#     # Select a subset of the dataset for testing
#     subset_size = 200  # Define the size of the subset
#     subset_indices = random.sample(range(len(full_dataset)), subset_size)
#     test_dataset = Subset(full_dataset, subset_indices)

#     # Create the test data loader
#     test_dataloader = DataLoader(test_dataset, batch_size=8, num_workers=4)

#     # Evaluate the model on the test set
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     avg_test_iou_tp, avg_test_iou_normal, avg_test_dice = evaluate_model(model, test_dataloader, device, n_classes)

#     print(f"Final Test IOU TP: {avg_test_iou_tp:.4f}")
#     print(f"Final Test IOU Normal: {avg_test_iou_normal:.4f}")
#     print(f"Final Test Dice Coefficient: {avg_test_dice:.4f}")

# if __name__ == "__main__":
#     main()



import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from unet import UNet
from archs.nested_unet import NestedUNet
from archs.deeplabv3 import DeepLabV3
from archs.segnet import SegNet
from coco_dataset import CocoDataset, transform
import random
from train import calculate_class_weights

# Set random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
# Ensure deterministic behavior for certain operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
    running_iou_tp = 0.0
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

            iou, iou_tp = calculate_iou(preds, masks, n_classes, dataset)
            dice = calculate_dice(preds, masks)
            

            running_iou += iou
            running_dice += dice
            running_iou_tp += iou_tp
            count += 1

    avg_iou = running_iou / count if count > 0 else float('nan')
    avg_iou_tp = running_iou_tp / count if count > 0 else float('nan')
    avg_dice = running_dice / count if count > 0 else float('nan')
    print(avg_iou_tp)
    return avg_iou, avg_iou_tp, avg_dice

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
    avg_iou, avg_iou_weighted, avg_test_dice = evaluate_model(model, test_dataloader, device, n_classes, dataset)

    print(f"Final Test IOU Weighted: {avg_iou:.4f}")
    print(f"Final Test IOU Normal: {avg_iou_weighted:.4f}")
    print(f"Final Test Dice Coefficient: {avg_test_dice:.4f}")
    return avg_iou, avg_iou_weighted, avg_test_dice

def main():
    name = 'models/segnet_xe_balanced_bs_32_seed_201.pth'
    isXE = True
    if 'xe' in name:
        isXE = True
    n_classes = 1  # Number of classes in your segmentation task
    if isXE:
        n_classes = 2
    model = SegNet(3, n_classes)

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
    avg_test_iou, avg_iou_tp, avg_test_dice = evaluate_model(model, test_dataloader, device, n_classes, test_dataset)

    print(f"Final Test mIOU: {avg_test_iou:.4f}")
    print(f"Final Test TP mIOU: {avg_iou_tp:.4f}")
    print(f"Final Test Dice Coefficient: {avg_test_dice:.4f}")

if __name__ == "__main__":
    main()
