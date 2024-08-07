# Leaf Hair Analysis

### Before you begin, you should have...
* Docker installed (If not, visit [here](https://docs.docker.com/engine/install/))
* Access to BioHPC

## 1. Build docker container: `./build.sh`
* Feel free to delete --no-cache so it will build faster next time

## 2. Edit `run.sh` for appropriate parameters

There are four different task commands:
1. `train`
2. `metrics`
3. `train_and_infer`
4. `ablation`

Change TASK to equal desired task. Each task has different parameters. Either
run with `python3 app.py $TASK --help` to see required and optional arguments
or see below:

# train
This task trains a model from the leaf data based on specified parameters

**Parameters**

| Option             | Description                                        | Required/Default Value                       |
|:--------------------|:----------------------------------------------------|:----------------------------------------------|
| `--leaf-images`      | Folder containing leaf images to train on         | `training_images`                           |
| `--annotations`      | COCO annotations file                             | `annotations/labelbox_coco.json`            |
| `--batch-size`     | Batch size for training                           | `32`                                        |
| `--epochs`         | Number of epochs for training                     | `100`                                       |
| `--learning-rate`  | Learning rate for training                        | `0.001`                                     |
| `--loss`           | Loss function to use for training                 | `dice`                                      |
| `--architecture`   | Model architecture                                | `DeepLabV3`                                 |
| `--balance`        | Whether to balance the class weights              | `True`                                      |
| `--gpu-index`      | Index of the GPU to use for training              | `1`                                         |


# infer
This task makes inferences (calcuates leaf hair analyses) using the model.

**Parameters**

| Option             | Description                                          | Required/Default Value                       |
|:--------------------|:------------------------------------------------------|:----------------------------------------------|
| `--image-dir`       | Directory containing images for inference           | *Required*                                    |
| `--model-path`      | Path to the trained model                           | `models/model-2.pth`                        |
| `--loss`            | Loss function used for training                     | `dice`                                      |
| `--architecture`    | Model architecture                                  | `DeepLabV3`                                 |
| `--results-folder`  | File to save the inference results                  | `results/hair_model_results.xlsx`            |
| `--use-model-1`     | Whether to use model from dataset 1 or 2            | `False`                                     |
| `--make-hair-mask`  | Whether to save reconstructed hair masks            | `False`                                     |

# metrics
This task calculates metrics to determine the quality of the model. The parameters MUST match the ones used for model training.

**Parameters**

| Option             | Description                                        | Required/Default Value                       |
|:--------------------|:----------------------------------------------------|:----------------------------------------------|
| `--model-path`      | Path to the trained model                         | *Required*                                    |
| `--leaf-images`     | Folder containing leaf images                      | `training_images`                           |
| `--annotations`     | COCO annotations file                             | `annotations/labelbox_coco.json`            |
| `--batch-size`      | Batch size for training                           | `32`                                        |
| `--epochs`          | Number of epochs for training                     | `100`                                       |
| `--loss`            | Loss function used for training                   | `dice`                                      |
| `--architecture`    | Model architecture                                | `DeepLabV3`                                 |
| `--balance`         | Whether to balance the class weights              | `True`                                      |
| `--gpu-index`       | Index of the GPU to use                           | `1`                                         |
| `--subset-size`     | Size of the subset for metrics calculation        | `200`                                       |

# train_and_infer
This task makes trains a model from the leaf data based on specified parameters and then
runs inferences using that generated model.

**Parameters**

| Option             | Description                                        | Required/Default Value                       |
|:--------------------|:----------------------------------------------------|:----------------------------------------------|
| `--image-dir`       | Directory containing images for inference          | *Required*                                    |
| `--leaf-images`     | Folder containing leaf images                      | `training_images`                           |
| `--annotations`     | COCO annotations file                             | `annotations/labelbox_coco.json`            |
| `--batch-size`      | Batch size for training                           | `32`                                        |
| `--epochs`          | Number of epochs for training                     | `10`                                        |
| `--learning-rate`   | Learning rate for training                        | `0.001`                                     |
| `--loss`            | Loss function to use for training                 | `dice`                                      |
| `--architecture`    | Model architecture                                | `DeepLabV3`                                 |
| `--balance`         | Whether to balance the class weights              | `True`                                      |
| `--gpu-index`       | Index of the GPU to use for training              | `1`                                         |
| `--results-folder`  | File to save the inference results                | `results/hair_model_results.xlsx`            |
| `--make-hair-mask`  | Whether to save reconstructed hair masks          | `False`                                     |

# ablation
This task runs an ablation to find the best model suited to the data. It looks at four
different architectures (DeepLabV3, UNet, UNet++, SegNet), three loss functions (Dice, DiceBCE, Cross Entropy),
three batch sizes (8, 16, 32), and balance (weighting classes inversely porportional to their frequencies).
The earlier ablation results can be found in `results/ablation_results.xlsx`.
Best Model: DeepLabV3, Dice, 32 batch size, balanced

**Parameters**

| Option             | Description                                        | Required/Default Value                       |
|:--------------------|:----------------------------------------------------|:----------------------------------------------|
| `--leaf-images`     | Folder containing leaf images                      | `training_images`                           |
| `--annotations`     | COCO annotations file                             | `annotations/labelbox_coco.json`            |
| `--excel-file`      | File to save ablation results                     | `results/ablation_results.xlsx`             |

## 3. Run docker: `./run.sh`
* For interactive mode, use `interactive.sh` and check app.py for usage

***Take a look at `example_run.sh` to see how your run.sh should be formatted.***
