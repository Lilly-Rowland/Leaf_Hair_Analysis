import os
import torch
import multiprocessing as mp
from coco_dataset import CocoDataset, transform
from train import run_train
from calculate_metrics import run_metrics
import pandas as pd
from postproccessing.confusion_matrix_evaluation import plot_confusion_matrix
import time
import logging

NUM_EPOCHS = 100
SEED = 555

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
BATCH_SIZE = [8,16,32]  # Batch sizes to experiment with


def run_experiment(dataset, experiment, gpu_index):
    arch, loss, balance, batch = experiment

    start_time = time.time()

    logging.info(f"Running experiment: Architecture={arch}, Loss={loss}, Balance={balance}, Batch Size={batch}")

    try:
        trained_model, avg_test_loss, name = run_train(dataset, loss=loss, arch=arch, balance=balance, seed=SEED, batch_size=batch, num_epochs=NUM_EPOCHS, gpu_index=gpu_index, ablation_run=True)
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        avg_iou, avg_iou_weighted, avg_test_dice, precision, recall, f1, total_conf_matrix = run_metrics(trained_model, dataset, arch, batch, loss=loss, gpu_index=gpu_index)
        plot_confusion_matrix(total_conf_matrix, classes=['Background', 'Leaf Hair'], name=f"results/confusion_matrices/{name}.png")

        result_entry = {
            'Architecture': ARCHITECTURES[arch],
            'Loss': LOSS_FUNCTIONS[loss],
            'Balance': 'Balanced' if balance else 'Unbalanced',
            'Batch Size': batch,
            'Average Test Loss': avg_test_loss,
            'Average IOU': float(avg_iou),
            'Average Weighted IOU': float(avg_iou_weighted),
            'Average Dice Coefficient': avg_test_dice,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Elapsed Time (seconds)': elapsed_time
        }
        logging.info(f"Finished Experiment: Architecture={arch}, Loss={loss}, Balance={balance}, Batch Size={batch}")
        return result_entry
    except Exception as e:
        logging.error(f"Experiment failed for: Architecture={arch}, Loss={loss}, Balance={balance}, Batch Size={batch}. Error: {str(e)}")
        return None

def run_ablation(img_dir, ann_file, excel_file = 'results/ablation_results.xlsx', isMain = False):
    
    # Setup logging configuration
    logging.basicConfig(filename='ablation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if not isMain:
        logger = logging.getLogger('ablation')
        logger.setLevel(logging.INFO)
    dataset = CocoDataset(img_dir, ann_file, transform=transform)
    results = mp.Manager().list()
    experiments = [(arch_name, loss_name, balance_option, batch_size)
                   for arch_name in ARCHITECTURES.keys()
                   for loss_name in LOSS_FUNCTIONS.keys()
                   for balance_option in BALANCE_OPTIONS
                   for batch_size in BATCH_SIZE]

    num_gpus = torch.cuda.device_count()
    logging.info(f"Available GPUs: {num_gpus}")

    experiment_queue = mp.Queue()
    for experiment in experiments:
        experiment_queue.put(experiment)

    def run_experiments(queue, gpu_index):
        while not queue.empty():
            experiment = queue.get()
            result = run_experiment(dataset, experiment, gpu_index)
            if result:
                results.append(result)

    processes = []
    for gpu_index in range(num_gpus):
        process = mp.Process(target=run_experiments, args=(experiment_queue, gpu_index))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    df = pd.DataFrame(list(results))
    df.to_excel(excel_file, index=False)
    logging.info(f"Excel file '{excel_file}' created successfully.")

if __name__ == "__main__":

    img_dir="Data"
    ann_file="Data/combined_coco.json"
    print("Script running in the background. Check 'ablation.log' for logs.")
    run_ablation(img_dir, ann_file, isMain = True)


#TO RUN: nohup python -u ablation.py > ablation.log &