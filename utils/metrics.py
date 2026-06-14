import torch
import numpy as np


def compute_detection_metrics(predictions, labels, threshold=0.5):
    predictions_binary = (predictions >= threshold).float()
    labels_binary = (labels >= 0.5).float()

    TP = ((predictions_binary == 1) & (labels_binary == 1)).sum().item()
    TN = ((predictions_binary == 0) & (labels_binary == 0)).sum().item()
    FP = ((predictions_binary == 1) & (labels_binary == 0)).sum().item()
    FN = ((predictions_binary == 0) & (labels_binary == 1)).sum().item()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    }


def compute_pd_pf(predictions, labels, threshold=0.5):
    predictions_binary = (predictions >= threshold).float()
    labels_binary = (labels >= 0.5).float()

    TP = ((predictions_binary == 1) & (labels_binary == 1)).sum().item()
    FN = ((predictions_binary == 0) & (labels_binary == 1)).sum().item()
    FP = ((predictions_binary == 1) & (labels_binary == 0)).sum().item()
    TN = ((predictions_binary == 0) & (labels_binary == 0)).sum().item()

    PD = TP / (TP + FN) if (TP + FN) > 0 else 0
    PF = FP / (FP + TN) if (FP + TN) > 0 else 0

    return PD, PF


def compute_pd_pf_per_scr(predictions, labels, threshold=0.5):
    predictions_binary = (predictions >= threshold).float()
    labels_binary = (labels >= 0.5).float()

    num_samples = labels_binary.size(0)
    pd_list = []
    pf_list = []

    for i in range(num_samples):
        pred_bin = predictions_binary[i]
        label_bin = labels_binary[i]

        TP = ((pred_bin == 1) & (label_bin == 1)).sum().item()
        FN = ((pred_bin == 0) & (label_bin == 1)).sum().item()
        FP = ((pred_bin == 1) & (label_bin == 0)).sum().item()
        TN = ((pred_bin == 0) & (label_bin == 0)).sum().item()

        PD = TP / (TP + FN) if (TP + FN) > 0 else 0
        PF = FP / (FP + TN) if (FP + TN) > 0 else 0

        pd_list.append(PD)
        pf_list.append(PF)

    return np.mean(pd_list), np.mean(pf_list)
