
    
import torch
import torch.nn.functional as F

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


