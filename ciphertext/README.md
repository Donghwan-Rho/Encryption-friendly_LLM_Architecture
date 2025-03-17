# Encryption-Friendly LLM Architecture (HE code) 

## Introduction
This folder is an implementation code for HE model in "Encryption-Friendly LLM Architecture"

## Prerequisites

- [Git](https://git-scm.com/)
- [Conda](https://docs.conda.io/en/latest/)

## Clone and build the project

```bash
conda env create -f conda/hellm-bert-env.yml
conda activate hellm-bert
cmake -S . -B build
cmake --build build -j
```

## Update Conda Environment

```bash
conda activate hellm-bert
conda env update --file ./conda/hellm-bert-env.yml  --prune
```

## Environment Variable Configuration

```bash
export HELLM_KEY_PATH=./key
```

## Description of running files for experiment.

1. Run `convert.py` to generate a container file for HE implementation with a pre-generated train/eval dataset running in plaintext, which is saved in `plaintext/fine-tuning_data/`. The file names are `taskname_eval_inputs(masks)` and `taskname_train_inputs(masks)` for the eval and train datasets, respectively. 
These files will be used in the next step and should set appropriate input data paths to generate weights for the HE model.

2. Run `convert2.cpp` to generate the HE version of the pre-trained weights. Use the pre-defined container path specified in the plaintext model and the designated output data folder. For more details, refer to the Fine-tuning explanation below.

3. Run either of the following two files:
- **Note** All the following implemenation codes need to work on 8 GPUs. You need to add `mpirun -np 8` to the first part of your running binary file.

- **3-1.** `/examples/backward-bert-multi.cpp`: Implementation code for fine-tuning.
- **3-2.** `/examples/bert-test.cpp`: Implementation code for evaluation.



## How to Set Required Parameters for Fine-Tuning 

Fine-tuning works in `/examples/backward-bert-multi.cpp`

1. Set the weight directory (`weight_pth`).
Specify the directory containing the model weights.

Example:
```bash
weight_pth = "./data_2ly_mrpc/"
```

2. Configure the number of GPUs (`num_gpu`).
Specify the number of GPUs to use

Default:
```bash
num_gpu = 8
```

##### Note: All experiments were performed on 8 GPUs, so the current implementation code is also 8 GPU-oriented. Experiments with less than 8 GPUs will require modifications to the code, especially the communication between GPU parts.

3. Set the batch size (`batch_size`).
Define the batch size for fine-tuning

Default:
```bash
batch_size = 2 # (Target batch size is 16; fine-tuning runs on 8 GPUs in parallel)
```
4. Specify the task name (`task`).
Use the following abbreviations for tasks:

```bash
task = "R" # RTE
task = "C" # COLA
task = "M" # MRPC
task = "S" # STSB
task = "T" # SST2
task = "Q" # QNLI
```

5. Set the total number of steps for one epoch (`one_epo_step`).
Set the number of steps per epoch for each task:

```bash
one_epo_step = 310 # RTE
one_epo_step = 458 # MRPC
one_epo_step = 718 # STSB
one_epo_step = 1068 # COLA
one_epo_step = 8418 # SST2
one_epo_step = 13092 # QNLI
```

6. Set the total number of train data (`num_data`).
Set the number of fine-tuning train data for each task:

```bash
num_data = 2490 # RTE
num_data = 3668 # MRPC
num_data = 5479 # STSB
num_data = 8551 # COLA
num_data = 67439 # SST2
num_data = 104743 # QNLI
```

7. Specify the pre-trained weight file (`container`), which has been set in the plaintext output file.

Provide the pre-trained weight file to be used.

Example
```bash
container = "converted_weights_mrpc.pth"
```

8. Set the label configuration (`labels`).
Provide the label mapping for the fine-tuning train dataset.
Each task's label is defined in `ModelArgs.hpp`

9. Configure the learning rate (`lr`).
Set the used learning rate during fine-tuning.
This parameter is defined in `ModelArgs.hpp`

10. Adjust the weight paths (`he_path, weight_path_`) in `HEMMer.hpp`.

```bash
"he_path" is used to save intermediated data for calculation.
"weight_path_" is used to save updated weights.
```

## How to get the metric score for each downstream task

1. set eval output file path in `metric.py`.

Example:
```bash
`file_path` = "mrpc_re_eval.txt" # This file should contain the eval output.
`label_true` = torch.load("./labels_mrpc_eval.pth").numpy().tolist () # The right-label file should be placed in the torch path.
```

2. Run `metric.py`
