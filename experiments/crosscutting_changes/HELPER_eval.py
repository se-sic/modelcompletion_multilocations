#todo if this is not working that way i need to make sure the test sets are the same
import json
from typing import Counter
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score
import torch


import os

from experiments.crosscutting_changes.HELPER_GENERIC import sort_graphs_chrono, split_train_val_test
from modules.graph.io import load_components_networkx


def load_test_data(output_path):

    all_test_node_pairs = {}
    all_test_labels = {}

    for subfolder in os.listdir(output_path):
        subfolder_path = os.path.join(output_path, subfolder)

        # Ensure it's a directory
        if os.path.isdir(subfolder_path):
            # Paths to train and test files
            test_file = os.path.join(subfolder_path, "dataset_pairs_test.pth")

            # Load test data if available
            if os.path.exists(test_file):
                test_data = torch.load(test_file)
                all_test_node_pairs[subfolder] = test_data["pairs"]
                all_test_labels[subfolder]=test_data["labels"]

    return all_test_node_pairs, all_test_labels

def convert_floats(obj):
   
    if isinstance(obj, dict):
        return {k: convert_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_floats(v) for v in obj)
    elif isinstance(obj, torch.Tensor):
        return convert_floats(obj.tolist() if obj.ndim > 0 else obj.item())
    elif isinstance(obj, np.ndarray):
        return convert_floats(obj.tolist())
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj



def _precision_at_k(rel_flags, denom):                    # changed
    """rel_flags = 0/1 list for top-k items; denom replaces k when < k."""     # changed
    return sum(rel_flags) / denom if denom else 0.0     

def _average_precision_at_k(sorted_labels, total_rel, k): # changed
    """AP@k for one query (macro style)."""               # changed
    if total_rel == 0:                                    # changed
        return 0.0                                        # changed
    hits = cum_prec = 0                                   # changed
    for i, rel in enumerate(sorted_labels[:k], 1):        # changed
        if rel:                                           # changed
            hits += 1                                     # changed
            cum_prec += hits / i                          # changed
    return cum_prec / min(k, total_rel)                   # changed

def compute_precison_recall_perfile(X, probabilities, labels, k ):
    X= X.cpu()
    #this computes the focus nodes, over which we will iterate So the whole expression gives you the set of unique 1536-dimensional vectors from the first "slot" of each sample.
    unique_keys = np.unique(X[:, 0, :], axis=0)
    # Get unique first-row values
    precision_list = []
    ap_list = []  
    ranked_suggestions_list = []
    probabilities = np.array(probabilities)
    labels = np.array(labels)
    total_true_positives=0
    total_predictions=0

    for key in unique_keys:
        # nothing important really So overall, it turns a NumPy 1D array key into a 2D PyTorch float tensor with shape [1, D].
        key = torch.tensor(key[np.newaxis, :], dtype=torch.float32)
        
        # get values of focus node: Each entry is True if the first slice of that sample equals key exactly, element-wise.
        matches = torch.all(X[:, 0, :] == key, dim=1)
        matching_indices = torch.nonzero(matches, as_tuple=True)[0].cpu().numpy()  # Convert to NumPy array
        matching_indices = matching_indices.astype(int).flatten()

        # given a focus node, get the probabilites of the other nodes changing with it 
        subset_probabilities = probabilities[matching_indices]
        # Sort in descending order
        ranked_indices = np.argsort(subset_probabilities)[::-1]
        # Apply sorting to second-row elements
        sorted_indices = matching_indices[ranked_indices]


       
       

        top_k_indices = sorted_indices[:k]
        ranked_suggestions = [(i, probabilities[i]) for i in top_k_indices]


        top_k_labels = [1 if labels[i] == 1 else 0 for i in top_k_indices]
        true_positives = sum(top_k_labels)
        total_true_positives += true_positives
      

         # Count how many "1"s are present in labels[matching_indices]
        total_positives = sum(labels[i] for i in matching_indices)
        adjusted_k = min(k, total_positives)
        total_predictions += adjusted_k

        if (adjusted_k > 0):
            precision_k = _precision_at_k(top_k_labels, adjusted_k)  
            #precision_k = true_positives / adjusted_k

            precision_list.append(precision_k)
            

    #batches_top5 = batchify_and_get_top5(probabilities, labels,batch_size=64)

    # Compute the average precision across all nodes
    avg_precision = np.mean(precision_list) if precision_list else -1
     
   
    #IMPORTANT: per example (one node) we take the precision_k and than average it again 
    return {
        "precision_avg": avg_precision,
        "precision_all": precision_list,
        "total_true_positives": total_true_positives,
        "total_predictions": total_predictions, 
        "unique_keys": len(unique_keys)
    }




def save_final_results(file_path, results, prefix):

    results = convert_floats(results)  # Ensure compatibility
    with open(file_path + prefix +"_results.json", "w") as f:
        json.dump(results, f, indent=4)  # Pretty-print for readability

def save_and_print_statistics(precision_topk, k, results,output_path,  all_outputs, all_labels, TRECHHOLD_DOWN, prefix):
    # print and compute the overall accurarcy 
    combined_outputs = [output for outputs in all_outputs.values() for output in outputs]
    combined_labels = [label for labels in all_labels.values() for label in labels]
    combined_outputs_np = np.array(combined_outputs)
    combined_labels_np = np.array(combined_labels)
    predictions = (combined_outputs_np > TRECHHOLD_DOWN).astype(int)

    acc = accuracy_score(combined_labels_np, predictions)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Label Distribution in Test Set: {Counter(combined_labels_np)}")


    # Print precision for files
    for file in precision_topk.keys():
        print(f"Top-{k} Suggestions for File {file}:")
        print(f"Precision@{k}: {precision_topk[file]:.4f}")
       # print(f"MAP score: {results[file]['map_k_avg']}")



    # print the overall, valid precision
    valid_precisions = [v for v in precision_topk.values() if v >= 0]
    avg_top5_precision = sum(valid_precisions) / len(valid_precisions) if valid_precisions else 0
    print(f"Average Precision {avg_top5_precision:.4f}")
    os.makedirs(output_path, exist_ok=True)
    
    save_final_results(output_path, results, prefix)


def get_test_graphs(input_path_data, input_path_splitting, name_sub_folders):
    test_graphs={}

    for folder_name in os.listdir(input_path_splitting):


        # Skip files in the input_path
        if not os.path.isdir(input_path_data + '/' + folder_name):
            continue

        input_dir = input_path_data + folder_name + name_sub_folders
        graphs = load_components_networkx(data_folder=input_dir, mark_filename=True)
        graphs = sort_graphs_chrono(graphs)


        split_index_train ,split_index_val = split_train_val_test(graphs)

        for i, graph in enumerate(graphs):

            # Initialize as a list for the dataset if it doesn't exist
            # add the test graphs to the dataset and skip them 
            if folder_name not in test_graphs:
                test_graphs[folder_name] = []

            if  i>= split_index_val:
                test_graphs[folder_name].append(graph)
    
    for folder, g_list in test_graphs.items():
        # if each graph has a stored label, count them …
        labels = [g.graph.get("label", "unknown") for g in g_list]
        label_counts = Counter(labels)
        print(f"{folder}: {label_counts}")

    return test_graphs

