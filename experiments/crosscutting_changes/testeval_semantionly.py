import os

import sys
import torch.nn.functional as F
import torch




current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Get the grandparent directory (parent of the parent)
grandparent_dir = os.path.dirname(parent_dir)

sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)

from experiments.crosscutting_changes.HELPER_eval import load_test_data, save_and_print_statistics
from experiments.crosscutting_changes.HELPER_eval import compute_precison_recall_perfile  #changed

# Set device
device = torch.device('mps' if torch.has_mps else 'cuda' if torch.cuda.is_available() else 'cpu')

TRECHHOLD_UP= 1
TRECHHOLD_DOWN =0.5
K=5


#print(output_path)
#neural_network_data -> all we have locally 
#output_path = "../output_dataset_label/TAGRETneural_network/"
#input_path_data = "../output_dataset_label/neural_network_data_small/"


input_path_data = "../output_dataset_label/embedding_data_refactored2"
output_path= "../results_final/"      






def compute_cosine_similarity(X):
     # Convert list of tuples to tensor: shape (N, 2, D)
    vec_pairs = [torch.stack([torch.tensor(tup[2], dtype=torch.float32),
                              torch.tensor(tup[4], dtype=torch.float32)]) for tup in X]
    vec_tensor = torch.stack(vec_pairs)  # shape (N, 2, D)

    vec1 = vec_tensor[:, 0, :]  # shape (N, D)
    vec2 = vec_tensor[:, 1, :]  # shape (N, D)

    similarities = F.cosine_similarity(vec1, vec2, dim=1)  # shape (N,)
    return vec_tensor, similarities

# Function to evaluate the model on the test set
def evaluate_model(input_path_data):
    # Load test data
    all_test_node_pairs, all_test_labels = load_test_data(input_path_data)
    all_outputs = {}
    all_labels = {}
    precision_topk = {}
    results = {}

    for file_path, X in all_test_node_pairs.items():
        labels = all_test_labels[file_path]
        tensors, similarities = compute_cosine_similarity(X)
        all_outputs[file_path] = similarities
        all_labels[file_path] = labels
        dict_prec = compute_precison_recall_perfile(tensors, similarities, labels, K)

        precision_topk[file_path]= dict_prec["precision_avg"]

        batch_meta = [
            (tup[0], tup[1], tup[3], -1)   # (graph_id, node_id, target_id, -1)
            for tup in X
        ]


        if file_path not in results:
            results[file_path] = {}
        

            
        results[file_path]["probabilities"] = similarities
        results[file_path]["labels"] = labels
        results[file_path]["meta"] = batch_meta
        results[file_path]["precision_avg"] = dict_prec["precision_avg"]
        results[file_path]["precision_all"]= dict_prec["precision_all"]
        results[file_path]["total_true_positives"]= dict_prec["total_true_positives"]
        results[file_path]["total_predictions"]= dict_prec["total_predictions"]

        print(f"folder_name {file_path}")
        print(f"Overall Precision: {dict_prec['precision_avg']:.4f} ({dict_prec['total_true_positives']}/{dict_prec['total_predictions']})")

        print(f"overall number graphs: should always be 1")
        print(f"number focus nodes: {dict_prec['unique_keys']}")


    save_and_print_statistics(precision_topk,K, results,output_path,  all_outputs, all_labels, TRECHHOLD_DOWN, "semantics")


   
if __name__ == "__main__":
    #we need this if we would like to train everything indiually 
    folderx = ['all']
   
    for x in folderx: 
        print("START")
        
        evaluate_model(input_path_data)
        print ("END_________________________________")
