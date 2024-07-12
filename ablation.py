import os
import torch
import multiprocessing as mp
from coco_dataset import CocoDataset, transform
from train import run_train  # Assuming your training function is in train.py
from tp_calculate_metrics import run_metrics

NUM_EPOCHS = 1
SEED = 201
# ARCHITECTURES = ["unet", "nested_unet", "deeplabv3", "segnet"]
# LOSS_FUNCTIONS = ['dice', 'xe', 'dicebce']
ARCHITECTURES = {
    'unet': "UNet",
    'nested_unet': "UNetPlusPlus",
    'deeplabv3': "DeepLabV3",
    'segnet': "SegNet"
}
LOSS_FUNCTIONS = {
    'xe': "CrossEntropyLoss",
    'dice': "DiceLoss",
    'dicebce': "DiceBCELoss"
}
BALANCE_OPTIONS = [False, True]  # Whether to balance class weights or not
BATCH_SIZE = [16]  # Batch sizes to experiment with
#BATCH_SIZE = [16, 32, 64]  # Batch sizes to experiment with


def run_experiment(dataset, experiment, gpu_index):
    # Unpack experiment parameters
    arch, loss, balance, batch = experiment

    # Run the experiment
    print(f"Running experiment for: Architecture={arch}, Loss={loss}, Balance={balance}, Batch Size={batch}")
    trained_model, avg_test_loss = run_train(dataset, loss=loss, arch=arch, balance=balance, seed=SEED, batch_size = batch, num_epochs=NUM_EPOCHS, gpu_index = gpu_index)

    # Run metrics
    avg_iou, avg_iou_weighted, avg_test_dice = run_metrics(trained_model, dataset, arch, batch, loss=loss, gpu_index = gpu_index)

    # Record results
    result_entry = {
        'Architecture': ARCHITECTURES[arch],
        'Loss': LOSS_FUNCTIONS[loss],
        'Balance': 'Balanced' if balance else 'Unbalanced',
        'Batch Size': batch,
        'Average Test Loss': avg_test_loss,
        'Average IOU': avg_iou,
        'Average Weighted IOU': avg_iou_weighted,
        'Average Dice Coefficient': avg_test_dice
        ##metrics hereeee'Validation Loss': val_losses[-1]  # Final validation loss after training
    }

    return result_entry

def run_ablation(dataset):
    results = []

    #definig experiments to run
    experiments = []
    for arch_name in ARCHITECTURES.keys():
        for loss_name in LOSS_FUNCTIONS.keys():
            for balance_option in BALANCE_OPTIONS:
                for batch_size in BATCH_SIZE:
                    experiments.append((arch_name, loss_name, balance_option, batch_size))
    
    # Determine available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

    # Create a multiprocessing Queue for experiments
    experiment_queue = mp.Queue()
    for experiment in experiments:
        experiment_queue.put(experiment)

    # Function to run experiments in a sequential manner
    def run_experiments(queue, gpu_index):
        while not queue.empty():
            experiment = queue.get()
            results.append(run_experiment(dataset, experiment, gpu_index))

    # Start processes to run experiments
    processes = []
    for gpu_index in range(num_gpus):
        process = mp.Process(target=run_experiments, args=(experiment_queue, gpu_index))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()


if __name__ == "__main__":

    dataset = CocoDataset(img_dir="Data", ann_file="Data/combined_coco.json", transform=transform)

    run_ablation(dataset)

    print("All experiments completed.")
