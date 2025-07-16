import os
import sys
import torch
from typing import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import time
import torch
import psutil
import os



current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Get the grandparent directory (parent of the parent)
grandparent_dir = os.path.dirname(parent_dir)

sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)
from experiments.crosscutting_changes.HELPER_GENERIC import make_output_folder
from experiments.crosscutting_changes.HELPER_neural_networks_dataloader import create_datasets
from experiments.crosscutting_changes.HELPER_configuration import BALANCE_METHOD, COMBINATION, LOSSFUNCTION, TRAIN_TEST_SPLIT


from experiments.crosscutting_changes.HELPER_neural_networks import  AttentionClassifier
from experiments.crosscutting_changes.HELPER_neural_networks_losses import focal_loss


device = (
    torch.device("cuda") if torch.cuda.is_available()
   # else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

print(f"Using device: {device}")  # CHANGED



random_seed=42
#how many folders for training, training data selection 
SAMPLE_SUBSET=100 

k_TRECHHOLD=5
TRECHHOLD= 0.5



def log_resource_usage(epoch, model, log_path, note=""):
    process = psutil.Process(os.getpid())

    cpu_usage = psutil.cpu_percent(interval=None)
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 ** 2)

    # Zeit auf GPU kann auf MPS nicht exakt gemessen werden, daher nur Hinweis
    device_name = next(model.parameters()).device

    log_message = f"[Epoch {epoch}] {note}\n" \
                  f"  CPU usage: {cpu_usage:.2f}%\n" \
                  f"  RAM usage: {memory_mb:.2f} MB\n" \
                  f"  Model device: {device_name}\n"

    if device_name.type == "cpu":
        log_message += " Model is on CPU – GPU not used!\n"
    elif device_name.type == "mps":
        log_message += " Model is using Apple MPS GPU\n"
    elif device_name.type == "cuda":
        log_message += " Model is using CUDA GPU\n"

    print(log_message)
    with open(log_path, "a") as f:
        f.write(log_message + "\n")


def cal_f1(predictions, targets, threshold=0.5):
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


def cal_precision(predictions, targets, threshold=0.5):
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

def recall(predictions, targets, threshold=0.5):
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


def top_k_precision(predictions, targets, k=k_TRECHHOLD):
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

def compute_loss(outputs, y_batch, pos_weight_factor,focal_loss_alpha, focal_loss_gamma , focal_loss_missclaf, pos_weight):
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
        criterion = nn.BCEWithLogitsLoss(reduction='none')  
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




#ACHTUNG_ nicht selbe top k precision wie in eval, hier nicht per graph 
# 3. Training Function
def train_model(model, train_loader, test_loader,output_neural_network, log_file_path,  epochs, lr,  pos_weight_factor,focal_loss_alpha, focal_loss_gamma , focal_loss_missclaf, all_embeddings):
    global INFLUENCE_FALSE_POSITIVE 
    pos_weight=compute_positive_weight(train_loader)
   
    #########define the optimzer ####################
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_map = 0.0 
    
    for epoch in range(epochs):

        model.train()

        epoch_loss = 0.0
        top_k_precisions = []
        top_k_aps=[]
        
        # Training Loop over batches
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch, all_embeddings).squeeze()

            loss=compute_loss(outputs=outputs, y_batch=y_batch, pos_weight_factor=pos_weight_factor,focal_loss_alpha=focal_loss_alpha, focal_loss_gamma=focal_loss_gamma , focal_loss_missclaf=focal_loss_missclaf, pos_weight=pos_weight)
   
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Evaluation Loop
        model.eval()
        test_preds, test_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch_original = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch, all_embeddings).squeeze()
                outputs = outputs.view(-1)
                
                outputs = torch.sigmoid(outputs)
                preds_original = (outputs > TRECHHOLD).float()

                if isinstance(preds_original, torch.Tensor):
                    preds = preds_original.cpu().tolist() 
                    y_batch = y_batch_original.cpu().tolist() 
                else:
                    preds = [preds_original] 
                    y_batch =[y_batch_original]

                test_preds.extend(preds)
                test_labels.extend(y_batch)

                 # Track top-5 precision
                precision = cal_precision(outputs, y_batch_original)
                ap_at_k=average_ap_at_k(outputs, y_batch_original)
                # precision = top_k_precision(outputs, y_batch_original)
                top_k_aps.append(ap_at_k)
                top_k_precisions.append(precision)


    
        avg_top_k_precision = sum(top_k_precisions) / len(top_k_precisions) if top_k_precisions else 0.0
        avg_ap_at_k = sum(top_k_aps) / len(top_k_aps) if top_k_aps else 0.0  # CHANGED

        if avg_ap_at_k > best_map:  # CHANGED
            best_map = avg_ap_at_k  # CHANGED


        acc = accuracy_score(test_labels, test_preds)
        log_resource_usage(epoch, model, log_file_path)
        epoch_message = f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}, Test Accuracy: {acc:.4f}, Top-5 Precision: {avg_top_k_precision:.4f}, MAP: {avg_ap_at_k:.4f}"  # CHANGED"
        label_distribution_message = f"Label Distribution: {Counter(test_preds)}"
        print(epoch_message)
        print(label_distribution_message)
        #with open(log_file_path, "w") as log_file:
         #   log_file.write(epoch_message + "\n")
          #  log_file.write(label_distribution_message + "\n\n")


        torch.save(model.state_dict(), f"{output_neural_network}/model_epoch_{epoch + 1}.pth")

    return best_map


def __main__(): 

    #LOCAL STUFF 
    input_path = "../output_dataset_label/dataset_node_embeddings_text-embedding-3-small-with-ids_small/diffgraphs/"
    #input_path = "../output_dataset_label/dataset_node_embeddings_text-embedding-3-small-with-ids-indivual-embeddings_small/diffgraphs/"
   
    directory_path_data =   "../output_dataset_label/embedding_data_refactored2"
    base_path = "../output_dataset_label/neural_network_data_small_output_refactored2/"
   



    LEARNING_RATE=0.003
    NUMBER_EPOCHS=1000
    BATCH_SIZE =1024

    #only for non-linear 
    HIDDEN_LAYERS = [2048, 128]


    LOSS_FOCAL_ALPHA =0.79
    LOSS_FOCAL_GAMMA = 3.0
    LOSS_FOCAL_MISCLASS_PENALTIY =6.0

    POS_WEIGHT_FACTOR=3.6


    output_neural_network = make_output_folder(base_path, TRAIN_TEST_SPLIT, BATCH_SIZE,HIDDEN_LAYERS, LOSSFUNCTION, LOSS_FOCAL_ALPHA, LOSS_FOCAL_GAMMA, LOSS_FOCAL_MISCLASS_PENALTIY, POS_WEIGHT_FACTOR, LEARNING_RATE, NUMBER_EPOCHS)
   
    os.makedirs(output_neural_network, exist_ok=True)
    output_neural_network= output_neural_network+ "/"

    # Open a log file for writing
    log_file_path = os.path.join(output_neural_network, "training_log.txt")


 
    print("loading data:")
    balanced_train_dataset, test_dataset, val_dataset, input_dim = create_datasets(input_path,directory_path_data, random_seed= random_seed, sample_subset=SAMPLE_SUBSET)
    # 4. Train and Compare Models
    print("Training Model:")


    train_loader = DataLoader(balanced_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


    # Initialize models
    #model = NonLinearModel(input_dim, HIDDEN_LAYERS).to(device)
    if COMBINATION ==  "ALL_EMBEDDINGS": 
        model = AttentionClassifier(embed_dim=1536*4, num_heads=16).to(device)
    else: 
        model = AttentionClassifier(embed_dim=1536, num_heads=16).to(device)

    train_model(model,  train_loader, test_loader,output_neural_network, log_file_path,  epochs=NUMBER_EPOCHS, lr=LEARNING_RATE,  pos_weight_factor=POS_WEIGHT_FACTOR,focal_loss_alpha=LOSS_FOCAL_ALPHA, focal_loss_gamma=LOSS_FOCAL_GAMMA , focal_loss_missclaf=LOSS_FOCAL_MISCLASS_PENALTIY, all_embeddings=COMBINATION)

if __name__ == "__main__":
     __main__()
