import os
import sys

import torch


current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Get the grandparent directory (parent of the parent)
grandparent_dir = os.path.dirname(parent_dir)

sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)
# For hyperparameter tuning you want a stable, threshold-independent metric
# otherwise your results depend too much on the arbitrary choice of threshold (0.5, 0.7, …).


# is dependent on threshold
def calculate_precision(predictions, targets, threshold=0.5):
    """
    Computes standard precision (binary classification).
    """
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    preds_bin = (predictions > threshold).float()

    true_positives = ((preds_bin == 1) & (targets == 1)).sum().item()
    predicted_positives = (preds_bin == 1).sum().item()

    if predicted_positives == 0:
        return 0.0  # avoid division by zero

    return true_positives / predicted_positives

# is dependent on threshold
def calculate_recall(predictions, targets, threshold=0.5):
    """
    Computes standard recall (binary classification).
    """
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    preds_bin = (predictions > threshold).float()

    true_positives = ((preds_bin == 1) & (targets == 1)).sum().item()
    actual_positives = (targets == 1).sum().item()

    if actual_positives == 0:
        return 0.0  # avoid division by zero

    return true_positives / actual_positives

# is dependent on threshold
def calculate_f1(predictions, targets, threshold=0.5):
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    preds_bin = (predictions > threshold).float()

    tp = ((preds_bin == 1) & (targets == 1)).sum().item()
    fp = ((preds_bin == 1) & (targets == 0)).sum().item()
    fn = ((preds_bin == 0) & (targets == 1)).sum().item()

    if tp + fp == 0 or tp + fn == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)

def soft_precision(predictions, targets):
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    # weighted true positives = probability mass assigned to true positives
    weighted_tp = (predictions * targets).sum().item()
    predicted_mass = predictions.sum().item()

    if predicted_mass == 0:
        return 0.0
    return weighted_tp / predicted_mass

# AP rewards models that rank true positives higher in the list 
# problem: rankning here may be different from what we actually test
# because in test we do it per graph and do not shuffle 

def average_ap_at_k(pred_scores, true_labels):
    """
    Compute AP@k for a single prediction (assumes pred_scores is a vector of scores, true_labels is binary 0/1).
    """
    sorted_indices = pred_scores.argsort(descending=True)
    true_sorted = true_labels[sorted_indices]

    hits = true_sorted == 1
    if hits.sum() == 0:
        return 0.0

    precisions = [(hits[:i+1].sum().item()) / (i+1) for i in range(len(hits))]
    return (hits.float() * torch.tensor(precisions)).sum().item() / hits.sum().item()



def top_k_precision(predictions, targets, k):
    """
    Computes precision for the top-k predictions closest to 1.
    """
    # Ensure predictions and targets are flattened
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    #effictive k, rename
    k = min(k, predictions.size(0))
    # Get the top-k indices
    top_k_pred, top_k_indices = torch.topk(predictions, k=k)

    # Ensure indices are on the same device as targets
    top_k_indices = top_k_indices.to(targets.device)

    # Extract the ground truth for the top-k predictions
    top_k_targets = targets[top_k_indices]
    #precision = (top_k_targets==top_k_pred).sum().item() / k  # Precision = TP / k
    precision = top_k_targets.sum().item() / k

    return precision