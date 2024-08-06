import argparse
from coco_dataset import CocoDataset, transform
from train import run_train
from make_inferences import get_inferences
from ablation import run_ablation
from calculate_metrics import run_metrics

def train(args):
    print("Training with the following options:")
    print(f"  --leaf-images: {args.leaf_images}")
    print(f"  --annotations: {args.annotations}")
    print(f"  --batch-size: {args.batch_size}")
    print(f"  --epochs: {args.epochs}")
    print(f"  --learning-rate: {args.learning_rate}")
    print(f"  --loss: {args.loss}")
    print(f"  --architecture: {args.architecture}")
    print(f"  --balance: {args.balance}")
    print(f"  --gpu-index: {args.gpu_index}")
    
    dataset = CocoDataset(img_dir=args.leaf_images, ann_file=args.annotations, transform=transform)
    name = run_train(
        dataset,
        args.loss,
        arch=args.architecture,
        balance=args.balance,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gpu_index=args.gpu_index
    )
    print(f"Model trained and saved as {name}")

def infer(args):
    print("Running inference with the following options:")
    print(f"  --model-path: {args.model_path}")
    print(f"  --image-dir: {args.image_dir}")
    print(f"  --results-folder: {args.results_folder}")
    print(f"  --use-model-1: {args.use_model_1}")
    print(f"  --make-hair-mask: {args.make_hair_mask}")
    
    get_inferences(args.model_path, args.image_dir, args.architecture, args.loss, args.results_folder, args.make_hair_mask)
    print(f"Inferences saved to {args.results_folder}")

def metrics(args):
    print("Calculating metrics with the following options:")
    print(f"  --model-path: {args.model_path}")
    print(f"  --leaf-images: {args.leaf_images}")
    print(f"  --annotations: {args.annotations}")
    print(f"  --batch-size: {args.batch_size}")
    print(f"  --epochs: {args.epochs}")
    print(f"  --loss: {args.loss}")
    print(f"  --architecture: {args.architecture}")
    print(f"  --balance: {args.balance}")
    print(f"  --gpu-index: {args.gpu_index}")
    print(f"  --subset-size: {args.subset_size}")

    dataset = CocoDataset(img_dir=args.leaf_images, ann_file=args.annotations, transform=transform)
    avg_iou, avg_iou_weighted, avg_test_dice, precision, recall, f1, total_conf_matrix = run_metrics(
        args.model_path,
        dataset,
        arch=args.architecture,
        batch=args.batch_size,
        loss=args.loss,
        subset_size=args.subset_size,
        gpu_index=args.gpu_index
    )
    
    output = {
        'Average IOU': float(avg_iou),
        'Average Weighted IOU': float(avg_iou_weighted),
        'Average Dice Coefficient': avg_test_dice,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }
    
    print(f"Metrics for {args.model_path}:\n{output}")

def train_and_infer(args):
    print("Training and running inference with the following options:")
    print(f"  --leaf-images: {args.leaf_images}")
    print(f"  --annotations: {args.annotations}")
    print(f"  --batch-size: {args.batch_size}")
    print(f"  --epochs: {args.epochs}")
    print(f"  --learning-rate: {args.learning_rate}")
    print(f"  --loss: {args.loss}")
    print(f"  --architecture: {args.architecture}")
    print(f"  --balance: {args.balance}")
    print(f"  --gpu-index: {args.gpu_index}")
    print(f"  --image-dir: {args.image_dir}")
    print(f"  --results-folder: {args.results_folder}")
    print(f"  --use-model-1: {args.use_model_1}")
    print(f"  --make-hair-mask: {args.make_hair_mask}")


    dataset = CocoDataset(img_dir=args.leaf_images, ann_file=args.annotations, transform=transform)

    model_path, _, _ = run_train(
        dataset,
        args.loss,
        arch=args.architecture,
        balance=args.balance,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gpu_index=args.gpu_index
    )
    get_inferences(model_path, args.image_dir, args.architecture, args.loss, args.results_folder, args.make_hair_mask)

def ablation(args):
    
    print("Running ablation with the following options:")
    print(f"  --leaf-images: {args.leaf_images}")
    print(f"  --annotations: {args.annotations}")
    print(f"  --excel-file: {args.excel_file}")

    run_ablation(args.leaf_images, args.annotations, args.excel_file)

    print(f"Ablation results saved to {args.excel_file}")

def main():
    parser = argparse.ArgumentParser(description="Run different tasks for your project")
    
    subparsers = parser.add_subparsers(dest="command")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--leaf-images", type=str, default="training_images", help="Folder containing leaf images")
    train_parser.add_argument("--annotations", type=str, default="labelbox_coco.json", help="COCO annotations file")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    train_parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for training")
    train_parser.add_argument("--loss", type=str, default="dice", help="Loss function to use for training")
    train_parser.add_argument("--architecture", type=str, default="DeepLabV3", help="Model architecture")
    train_parser.add_argument("--balance", type=bool, default=True, help="Whether to balance the class weights")
    train_parser.add_argument("--gpu-index", type=int, default=1, help="Index of the GPU to use for training")
    train_parser.set_defaults(func=train)

    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--model-path", type=str, default='models/labelbox_data_DeepLabV3_dice_balanced_bs_64_seed_201.pth', help="Path to the trained model")
    infer_parser.add_argument("--image-dir", type=str, required=True, help="Directory containing images for inference")
    infer_parser.add_argument("--loss", type=str, default="dice", help="Loss function used for training")
    infer_parser.add_argument("--architecture", type=str, default="DeepLabV3", help="Model architecture")
    infer_parser.add_argument("--results-folder", type=str, default="hair_model_results.xlsx", help="File to save the inference results")
    infer_parser.add_argument("--use-model-1", type=bool, default=False, help="Whether to use model from dataset 1 or 2")
    infer_parser.add_argument("--save-hair-masks", type=bool, default=False, help="Whether to save reconstructed hair masks")
    infer_parser.set_defaults(func=infer)

    # Metrics command
    metrics_parser = subparsers.add_parser("metrics", help="Calculate metrics")
    metrics_parser.add_argument("--model-path", type=str, default='models/labelbox_data_DeepLabV3_dice_balanced_bs_64_seed_201.pth', help="Path to the trained model")
    metrics_parser.add_argument("--leaf-images", type=str, default="training_images", help="Folder containing leaf images")
    metrics_parser.add_argument("--annotations", type=str, default="labelbox_coco.json", help="COCO annotations file")
    metrics_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    metrics_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    metrics_parser.add_argument("--loss", type=str, default="dice", help="Loss function used for training")
    metrics_parser.add_argument("--architecture", type=str, default="DeepLabV3", help="Model architecture")
    metrics_parser.add_argument("--balance", type=bool, default=True, help="Whether to balance the class weights")
    metrics_parser.add_argument("--gpu-index", type=int, default=1, help="Index of the GPU to use")
    metrics_parser.add_argument("--subset-size", type=int, default=200, help="Size of the subset for metrics calculation")
    metrics_parser.set_defaults(func=metrics)

    # Train and infer command
    train_and_infer_parser = subparsers.add_parser("train_and_infer", help="Train the model and then run inference")
    train_and_infer_parser.add_argument("--leaf-images", type=str, default="training_images", help="Folder containing leaf images")
    train_and_infer_parser.add_argument("--annotations", type=str, default="labelbox_coco.json", help="COCO annotations file")
    train_and_infer_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    train_and_infer_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    train_and_infer_parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for training")
    train_and_infer_parser.add_argument("--loss", type=str, default="dice", help="Loss function to use for training")
    train_and_infer_parser.add_argument("--architecture", type=str, default="DeepLabV3", help="Model architecture")
    train_and_infer_parser.add_argument("--balance", type=bool, default=True, help="Whether to balance the class weights")
    train_and_infer_parser.add_argument("--gpu-index", type=int, default=1, help="Index of the GPU to use for training")
    train_and_infer_parser.add_argument("--image-dir", type=str, required=True, help="Directory containing images for inference")
    train_and_infer_parser.add_argument("--results-folder", type=str, default="hair_model_results.xlsx", help="File to save the inference results")
    train_and_infer_parser.add_argument("--use-model-1", type=bool, default=False, help="Whether to use model from dataset 1 or 2")
    train_and_infer_parser.add_argument("--save-hair-masks", type=bool, default=False, help="Whether to save reconstructed hair masks")
    train_and_infer_parser.set_defaults(func=train_and_infer)

    # Ablation command
    ablation_parser = subparsers.add_parser("ablation", help="Run ablation")
    ablation_parser.add_argument("--leaf-images", type=str, default="training_images", help="Folder containing leaf images")
    ablation_parser.add_argument("--annotations", type=str, default="labelbox_coco.json", help="COCO annotations file")
    ablation_parser.add_argument("--excel-file", type=str, default="ablation_results.xlsx", help="File to save ablation results")
    ablation_parser.set_defaults(func=ablation)

    args = parser.parse_args()
    args.func(args)
    

if __name__ == "__main__":
    main()
