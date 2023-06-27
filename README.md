# xnap-project-matcad_grup_3 Repository

This repository contains the source code for a project divided into two main folders: `coral-cnn-master` and `AgePrediction`.

## coral-cnn-master
The `coral-cnn-master` folder serves as the starting point for the project. It includes several important subfolders:

- `model-code`: This directory houses nine executable files, each responsible for training a model over a span of 200 epochs.

- `single-image-prediction_w-pretrained-models`: Here, you can find files that enable the prediction of the age of a single image using pretrained models.

- `experiment-logs`: This folder stores the results of the trained models, including predictions, losses, and model parameters.

For more information about the files in the `coral-cnn-master` folder or to access the download links, please refer to the `coral-cnn-master README`.

## AgePrediction
The `AgePrediction` folder contains the following structure:

- `models` directory: This directory consists of the `_init_.py` and `models.py` files. They define classes and functions related to the models used in the project. Many of these classes and functions are generalizations of the models found in `coral-cnn-master`, with minor enhancements.

- `utils` directory: Here, you will find the `_init_.py` and `utils.py` files, which define various functions used in the execution files. Most of these functions originate from the starting point code.

- `sortides` directory: This folder serves as the destination for files generated during executions. The `--outpath` parameter should always be specified as a relative path to this folder.

- `main.py`: This is the main executable file, encompassing the project's overarching ideas. It allows for modifications to the dataset or loss using flag parameters, initialization of a pre-trained model for improvement, and predictions using the test dataset.


**Example**

The following code trains `coral` on the `CACD` dataset:

```bash
python main.py --cuda 0 --seed 1
--outpath ./AgePrediction/sortides/predictions-cacd_coral
--dataset cacd
--loss coral
--starting_params 0
--state_dict_path /home/xnmaster/projecte/AgePrediction/sortides/cacd-pretrained/cacd-coral__seed1/best_model.pt
```

- `--cuda <int>`: The CUDA device number of the GPU to be used for training 
(`--cuda 0` refers to the 1st GPU).
- `--seed <int>`: Integer for the random seed; used for training set shuffling and
the model weight initialization (note that CUDA convolutions are not fully deterministic).
- `--outpath <directory>`: Path for saving the training log (`training.log`) 
and the parameters of the trained model (`model.pt`). 
- `--dataset <str>`: Flag that indicates the dataset to use (CACD, AFAD).
- `--loss <str>`: Flag that indicates the loss function to use (ce, coral, ordinal).
- `--starting_params <int>`: Integer to indicate whether or not to use the best parameters of a pretrained model.
- `--state_dict_path <directory>`: Path which contains the pretrained model file (best_model.pt).

