import os
import sys
import torch
from typing import Counter
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import time
import torch
import os



current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Get the grandparent directory (parent of the parent)
grandparent_dir = os.path.dirname(parent_dir)

sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)
from experiments.crosscutting_changes.helper.HELPER_GENERIC import make_output_folder
from experiments.crosscutting_changes.helper.HELPER_neural_networks_dataloader import create_datasets
from experiments.crosscutting_changes.helper.HELPER_configuration import COMBINATION, LOSSFUNCTION, TRAIN_TEST_SPLIT
from experiments.crosscutting_changes.helper.HELPER_objectives import average_ap_at_k, calculate_precision
from experiments.crosscutting_changes.helper.HELPER_neural_networks import  AttentionClassifier
from experiments.crosscutting_changes.helper.HELPER_neural_networks_losses import compute_loss, compute_positive_weight


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


#ACHTUNG_ nicht selbe top k precision wie in eval, hier nicht per graph 
# 3. Training Function
def train_model(model, train_loader, test_loader,output_neural_network, log_file_path,  epochs, lr,  pos_weight_factor,focal_loss_alpha, focal_loss_gamma , focal_loss_missclaf, all_embeddings):
    global INFLUENCE_FALSE_POSITIVE 
    pos_weight=compute_positive_weight(train_loader)
   
    #########define the optimzer ####################
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_epoch_for_hyperparametertuning = 0.0 
    
    for epoch in range(epochs):

        model.train()

        epoch_loss = 0.0
        top_k_precisions = []
        top_k_aps=[]
        
        # Training Loop over batches
        for X_batch, y_batch_ground_truth_cpu in train_loader:
            X_batch, y_batch_ground_truth_cpu = X_batch.to(device), y_batch_ground_truth_cpu.to(device)
            optimizer.zero_grad()
            outputs_probabilities = model(X_batch, all_embeddings).squeeze()

            loss=compute_loss(outputs=outputs_probabilities, y_batch=y_batch_ground_truth_cpu, pos_weight_factor=pos_weight_factor,focal_loss_alpha=focal_loss_alpha, focal_loss_gamma=focal_loss_gamma , focal_loss_missclaf=focal_loss_missclaf, pos_weight=pos_weight, device=device)
   
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Evaluation Loop
        model.eval()
        test_preds, test_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch_ground_truth_cpu in test_loader:
                X_batch, y_batch_ground_truth = X_batch.to(device), y_batch_ground_truth_cpu.to(device)
                outputs_probabilities = model(X_batch, all_embeddings).squeeze()
                outputs_probabilities = outputs_probabilities.view(-1)
                
                outputs_probabilities = torch.sigmoid(outputs_probabilities)
                prediction_int = (outputs_probabilities > TRECHHOLD).float()

                if isinstance(prediction_int, torch.Tensor):
                    prediction_int_cpu = prediction_int.cpu().tolist() 
                    y_batch_ground_truth_cpu = y_batch_ground_truth.cpu().tolist() 
                else:
                    prediction_int_cpu = [prediction_int] 
                    y_batch_ground_truth_cpu =[y_batch_ground_truth]

                test_preds.extend(prediction_int_cpu)
                test_labels.extend(y_batch_ground_truth_cpu)

                precision = calculate_precision(outputs_probabilities, y_batch_ground_truth)
                ap_at_k = average_ap_at_k(outputs_probabilities, y_batch_ground_truth)
                top_k_aps.append(ap_at_k)
                top_k_precisions.append(precision)


        # compute the overall test metrics for this epoch 
        avg_top_k_precision = sum(top_k_precisions) / len(top_k_precisions) if top_k_precisions else 0.0
        avg_ap_at_k = sum(top_k_aps) / len(top_k_aps) if top_k_aps else 0.0 
        acc = accuracy_score(test_labels, test_preds)

        # keep track of the overall best value 
        if avg_ap_at_k > best_epoch_for_hyperparametertuning: 
            best_epoch_for_hyperparametertuning = avg_ap_at_k  

        epoch_message = f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}, Test Accuracy: {acc:.4f}, Top-5 Precision: {avg_top_k_precision:.4f}, MAP: {avg_ap_at_k:.4f}"
        label_distribution_message = f"Label Distribution: {Counter(test_preds)}"
        print(epoch_message)
        print(label_distribution_message)
        torch.save(model.state_dict(), f"{output_neural_network}/model_epoch_{epoch + 1}.pth")

    return best_epoch_for_hyperparametertuning


def __main__(): 

    #LOCAL STUFF 
    input_path = "../output_dataset_label/dataset_node_embeddings_text-embedding-3-small-with-ids_small/diffgraphs/"
    #input_path = "../output_dataset_label/dataset_node_embeddings_text-embedding-3-small-with-ids-indivual-embeddings_small/diffgraphs/"
    #input_path = "/scratch/TODO/CrossCutting/dataset_node_embeddings_text-embedding-3-small-with-ids/diffgraphs/"
    
    #directory_path_data =  "/scratch/TODO/CrossCutting/neuralnetworkpairs/embedding_data_new_branch_indivudalemb/"
    directory_path_data =   "../output_dataset_label/embedding_data_refactored2"
    directory_path_data = "../resultsmajorrevision/datasets_major_revision_sample_10fold/fold_1__eclipse.e4!!bundles_org.eclipse.e4.ui.model.workbench_model_UIElements.ecore/"
    base_path = "../output_dataset_label/neural_network_data_small_output_refactored2/"
    #base_path =  "/scratch/TODO/CrossCutting/neuralnetworksgpu/"

   # input_path = "/scratch/TODO/dataset_node_embeddings_text-embedding-3-small-with-ids/diffgraphs/"
    #directory_path_data =  "/scratch/TODO/CrossCutting/neuralnetworkpairs_gpu/embedding_data_new_branch_ONLYONE_mergedemb/"
    #base_path =  "/scratch/TODO/CrossCutting/outputneuralnetwork/final_NN_ONLYONE_mergedemb_limit100000/"




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
