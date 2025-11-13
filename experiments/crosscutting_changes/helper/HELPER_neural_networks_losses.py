
    
from typing import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.crosscutting_changes.helper.HELPER_configuration import BALANCE_METHOD, LOSSFUNCTION

def focal_loss(logits, targets, base_loss, alpha, gamma,misclass_penality, reduction='mean'):
  
    # Convert raw logits -> probabilities
    probs = torch.sigmoid(logits)  

    # Binary Cross Entropy for each sample, no reduction yet
    bce = base_loss(logits, targets)

    # pt is probability of the *true* class
    # if y=1, pt = p; if y=0, pt = 1-p
    pt = probs * targets + (1 - probs) * (1 - targets)

    # focal weight: (1-pt)^gamma
    focal_weight = (1.0 - pt).pow(gamma)

    #this is if datat is imbalanced 
    alpha_factor = alpha * targets + (1.0 - alpha) * (1.0 - targets)

    # apply focal weight to the BCE
    # Strong penalty for incorrect high-probability zeros
    misclass_penalty_factor = (targets == 0) * probs * misclass_penality + 1.0  # Boost wrong high-prob zero cases

    # Compute final focal loss,
    fl = alpha_factor * focal_weight * bce * misclass_penalty_factor  
    #fl = alpha_factor * focal_weight * bce #10 #contant factor 

    # optional alpha weighting (commonly used if positives are rare)

    if reduction == 'mean':
        return fl.mean()

    else:
        return fl


def compute_loss(outputs, y_batch, pos_weight_factor,focal_loss_alpha, focal_loss_gamma , focal_loss_missclaf, pos_weight, device):
    #original BCE loss for undersampling
    assert BALANCE_METHOD != "undersampling" or LOSSFUNCTION == "BCELoss"

    if LOSSFUNCTION == "BCEWithLogitsLoss":
        pos_weight_tensor = torch.tensor([pos_weight_factor * pos_weight], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        loss = criterion(outputs, y_batch)
    elif LOSSFUNCTION == "BCELoss":
        #SIGMOID required before
        criterion = nn.BCELoss()  # Original loss function for undersampling
        loss = criterion(outputs, y_batch)
    #elif LOSSFUNCTION =="Loss_Top5":
        #loss = top_k_penalized_loss(outputs, y_batch, k=k_TRECHHOLD, base_loss=criterion)

    elif LOSSFUNCTION == "focalLoss":
        criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([pos_weight_factor * pos_weight], device=device))
        loss= focal_loss(outputs, y_batch, alpha=focal_loss_alpha, gamma=focal_loss_gamma, misclass_penality=focal_loss_missclaf, base_loss=criterion)

    return loss

def compute_positive_weight(train_loader):
    pos_weight= 1
    if BALANCE_METHOD == "weighted_loss":
        # Assume "weighted_loss"
        # Compute class weights for weighted loss
        train_labels = torch.cat([y.unsqueeze(0) if y.dim() == 0 else y for _, y in train_loader.dataset])
        class_counts = Counter(train_labels.tolist())
        pos_count = class_counts[1]  # according to docu this is the correct way
        neg_count = class_counts[0]  # Number of negative samples
        pos_weight = torch.tensor([ neg_count / pos_count])
        #class_weights = {cls: total_samples / count for cls, count in class_counts.items()} 

        #print(f"Class Weights: {pos_weight}")  
        # Convert class weights to tensor
        #weights = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32).to(device)  
       # criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT_FACTOR * pos_weight).to(device) 
    return pos_weight


