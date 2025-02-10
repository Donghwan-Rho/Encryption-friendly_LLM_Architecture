""" This is a source file for computing downstream task scores."""

import torch
import sys
import math
import numpy as np
from safetensors import safe_open

from sklearn.metrics import confusion_matrix


def sort_data_by_blocks(file_path, block_size=8):

    current_block = []
    blocks = []

    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            # Skip lines before line 17
            # Because the first 16 lines represent unrelated informations.
            if line_number < 17:
                continue

            cleaned_line = line.strip().rstrip(',')
            if not cleaned_line:
                continue

            try:
                line_data = list(map(float, cleaned_line.split(',')))
                current_block.append(line_data)
            except ValueError:
                print(f"Skipping invalid line: {line}")
                continue

            if len(current_block) == block_size:
                sorted_block = sorted(current_block, key=lambda x: x[0])
                blocks.extend(sorted_block)
                current_block = []

        if current_block:
            sorted_block = sorted(current_block, key=lambda x: x[0])
            blocks.extend(sorted_block)

    return blocks


def evaluate_sorted_data(blocks):
    he_list = []

    for line in blocks:
        if len(line) < 3:
            print(f"Skipping invalid line with insufficient values: {line}")
            continue

        value1 = line[1]
        value2 = line[2]

        if value1 > value2:
            he_list.append(0)
        else:
            he_list.append(1)

    return he_list

# Modify file_path with the eval output data file.
file_path = "mrpc_re_eval.txt"
sorted_blocks = sort_data_by_blocks(file_path)
he_list = evaluate_sorted_data(sorted_blocks)


# Need to load an appropriate label .pth file
# ```
# For example, "./labels_mrpc_eval.pth" denotes label file containig mrpc_eval set.
# ```



label_true = torch.load("./labels_mrpc_eval.pth").numpy().tolist()
label_he_pred = he_list


#  Metrics for each benchmark downstream task
#
# * RTE: ACC
# * MRPC: F1
# * COLA: MCC
# * STS-B: Pearson
# * SST-2: ACC
# * QNLI: ACC
#

# Calcuate F1 score
def calculate_f1_score(y_true, y_pred):
    TP = sum((y_true[i] == 1 and y_pred[i] == 1) for i in range(len(y_true)))
    FP = sum((y_true[i] == 0 and y_pred[i] == 1) for i in range(len(y_true)))
    FN = sum((y_true[i] == 1 and y_pred[i] == 0) for i in range(len(y_true)))

    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score

# Calculate MCC
def calculate_mcc(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    if denominator == 0:
        return 0

    return numerator / denominator

# Calculate ACC
def calculate_acc(y_true, y_pred) :
    same_value_count = 0
    list_length = len(y_true)

    for i in range(list_length):
        if y_true[i] == y_pred[i]:
            same_value_count += 1

    same_value_ratio = same_value_count / list_length

    return same_value_ratio

# Calculate Pearson corr.
def pearson_correlation(x, y):

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    numerator = np.sum((x - mean_x) * (y - mean_y))

    denominator = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))

    r = numerator / denominator

    return r


mcc = calculate_mcc(label_true, label_he_pred)
print(f"Matthews Correlation Coefficient: {mcc: .4f}")

f1 = calculate_f1_score(label_true, label_he_pred)
print(f"F1 score: {f1: .4f}")

acc = calculate_acc(label_true, label_he_pred)
print(f"Acc: {acc:.4f}")

pearson = pearson_correlation(label_true, label_he_pred)
print(f"pearson: {pearson:.4f}")
