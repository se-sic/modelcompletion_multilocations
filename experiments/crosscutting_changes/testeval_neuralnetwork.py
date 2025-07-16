import os

import sys
import torch
from collections import Counter





current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Get the grandparent directory (parent of the parent)
grandparent_dir = os.path.dirname(parent_dir)

sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)

from experiments.crosscutting_changes.HELPER_configuration import COMBINATION
from experiments.crosscutting_changes.HELPER_eval import compute_precison_recall_perfile, load_test_data, save_and_print_statistics
from modules.graph.io import load_components_networkx
from experiments.crosscutting_changes.HELPER_neural_networks import  NonLinearModel, AttentionClassifier

# Set device
device = torch.device('mps' if torch.has_mps else 'cuda' if torch.cuda.is_available() else 'cpu')

TRECHHOLD_UP= 1
TRECHHOLD_DOWN =0.5
K=5
name_sub_folders= "/default/"
# Function to evaluate the model on the test set
def evaluate_model(input_path_graphs, input_path_test_data, output_neural_network,output_path,  EPOCH):
    # Load test data
    all_test_node_pairs, all_test_labels = load_test_data(input_path_test_data)
    allgraphs={}
    # Keep test data in a mappable format
    test_file_map = {
        file_path: (pairs, labels)
        for file_path, (pairs, labels) in zip(all_test_node_pairs.keys(), zip(all_test_node_pairs.values(), all_test_labels.values()))
    }

    #input_dim = list(all_test_node_pairs.values())[0].shape[1] * list(all_test_node_pairs.values())[0].shape[2]

    
    #get the graphs for computing graph statistics 
    for folder_name in os.listdir(input_path_graphs):
        # Generate name for the output folder
        if not os.path.isdir(input_path_graphs + '/' + folder_name):
                continue
        input_dir = input_path_graphs + '/' + folder_name + name_sub_folders
        
        # do the dataset splitting
        graphs = load_components_networkx(data_folder=input_dir, mark_filename=True)
        allgraphs[folder_name] = {}
        for graph in graphs:
            allgraphs[folder_name][int(graph.diff_id.split("_")[1].split(".")[0])] = graph
            G_undirected = graph.to_undirected()
            if not nx.is_connected(G_undirected):
                print(f"[WARN] Graph  in {folder_name} is NOT connected.")
        

        
    #model = NonLinearModel(1536,[2048, 128] ).to(device)
    model = AttentionClassifier(embed_dim=1536, num_heads=16).to(device)
    model_file = os.path.join(output_neural_network, EPOCH)
    model_file = output_neural_network+ EPOCH
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    # Evaluate the model by iterating over file names
    all_outputs = {}
    all_meta = {}
    all_labels = {}
    precision_topk = {}
    results={}

    with torch.no_grad():
        for file_path, (X, y) in test_file_map.items():

            # new 
            batch_inputs = []
            batch_meta = []

            for tup in X:
                vec1 = tup[2]  # e.g. torch.Tensor shape (1536,)
                vec2 = tup[4]
                vec1 = torch.tensor(tup[2], dtype=torch.float32)
                vec2 = torch.tensor(tup[4], dtype=torch.float32)
                input_pair = torch.stack([vec1, vec2])  # shape (2, 1536)
                batch_inputs.append(input_pair)
                batch_meta.append((tup[0], tup[1], tup[3]))  # keep for mapping

            X_tensor = torch.stack(batch_inputs).to(device)  # shape (N, 2, 1536)
            y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

            outputs = model(X_tensor, more_embeddings= COMBINATION).squeeze()  # shape (N,)
            outputs= torch.sigmoid(outputs)


            #X, y = X.to(device), y.to(device)
            #outputs = model(X).squeeze()
            

            probabilities = outputs.cpu().tolist()
            labels =  y_tensor.cpu().tolist()

            batch_meta = compute_shortest_distance(batch_meta, file_path, allgraphs)
            target_radius_dependence_1, target_radius_dependence_2, target_radius_dependence_3 = compute_labels_target_distance(batch_meta, file_path, allgraphs,  labels )

            all_outputs[file_path] = probabilities
            all_labels[file_path] = labels
            all_meta [file_path] = batch_meta 
          

            label_counts = Counter(labels)
            label_counts = Counter(labels)

            
            print (label_counts)

            dict_prec = compute_precison_recall_perfile(X_tensor, probabilities, labels, k=K)
           
            if file_path not in results:
                results[file_path] = {}
            precision_topk[file_path]= dict_prec ["precision_avg"]
            
            results[file_path]["probabilities"] = probabilities
            results[file_path]["labels"] = labels
            results[file_path]["meta"] = batch_meta
            results[file_path]["precision_avg"] = dict_prec["precision_avg"]
            results[file_path]["precision_all"]= dict_prec["precision_all"]
            results[file_path]["total_true_positives"]= dict_prec["total_true_positives"]
            results[file_path]["total_predictions_radius_1"] = target_radius_dependence_1
            results[file_path]["total_predictions_radius_2"] = target_radius_dependence_2
            results[file_path]["total_predictions_radius_3"] = target_radius_dependence_3
            #results[file_path]["map_k_avg"]=      dict_prec [ "map_k_avg"]  # changed (macro MAP@k for this file)
            #results[file_path]["map_all"]= dict_prec ["map_all"]



        save_and_print_statistics(precision_topk,K, results,output_path,  all_outputs, all_labels, TRECHHOLD_DOWN, "neuralnetwork")

import networkx as nx
import networkx as nx
from collections import defaultdict

def compute_labels_target_distance(batch_meta, foldername, graphs, labels):
    """
    Parameters
    ----------
    batch_meta : iterable of (graph_id, node1, node2)
    foldername : str
    graphs     : dict[str, dict[graph_id, nx.Graph]]
    labels     : iterable of 0/1, same length/order as batch_meta

    Returns
    -------
    list[dict] – one dict per (graph_id, node1, node2) triple
    """
    # 1) collect, for every (graph_id, node1), the set of node2 with label==1
    pos_targets = defaultdict(set)          # key = (graph_id, node1)
    for (graph_id, node1, node2, dist), lab in zip(batch_meta, labels):
        if lab == 1:
            pos_targets[(graph_id, node1)].add(str(node2))

    # 2) pre-compute distances for each unique (graph_id, node1)
    dist_cache = {}                         # same key as above
    for (graph_id, node1), tgt_nodes in pos_targets.items():
        G = graphs[foldername][graph_id]
        if tgt_nodes:                       # multi-source BFS up to radius 3
            dist_cache[(graph_id, node1)] = nx.multi_source_dijkstra_path_length(
                G, tgt_nodes, cutoff=3
            )
        else:                               # no positive targets ⇒ empty sets
            dist_cache[(graph_id, node1)] = {}

    # 3) assemble per-sample results
    r1, r2, r3 = [], [], []
    for (graph_id, node1, node2, dist) in batch_meta:
        d = dist_cache[(graph_id, node1)].get(str(node2), 99) # 99 ⇒ “> 3 hops / unreachable”
        r1.append(int(d <= 1))
        r2.append(int(d <= 2))
        r3.append(int(d <= 3))

    return r1, r2, r3

   



def compute_shortest_distance(batch_meta, foldername, graphs):
    result = []
    for meta in batch_meta:
        graph_id, node1, node2 = meta[0], meta[1], meta[2]
        G = graphs[foldername][graph_id]
        G_undirected = G.to_undirected()

        try:
            # There's no cutoff where it gives up and says "too far"; it will return the real distance if a path exists.
            dist = nx.shortest_path_length(G_undirected, source=str(node1), target=str(node2))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            dist = float('inf')  # or use -1 if preferred
        result.append((*meta, dist))
    return result


if __name__ == "__main__":
    folderx = ['all']
    EPOCHS = ["model_epoch_2.pth"]
    for i, x in enumerate(folderx):
        
        print("START")
       
        input_path_graphs = "../output_dataset_label/dataset_node_embeddings_text-embedding-3-small-with-ids_small/diffgraphs/"
       # input_path_graphs="../output_dataset_label/embedding_data_refactored2"
      
        input_path_test_data = "../output_dataset_label/embedding_data_refactored2"
       # input_path_test_data = "../output_dataset_label/embedding_data_refactored2"
        input_neural_network =  "../output_dataset_label/neural_network_data_small_output_refactored/nnout_split-TRAINVALTEST_batch-1024_layers-2048-128_loss-BCEWithLogitsLoss_alpha-0.79_gamma-3.0_mispen-6.0_posw-3.6_lr-0.003_epochs-1000/"
        
        output= "../results_final/"




        EPOCH= EPOCHS[i]
        #print(output_path)
        #print(EPOCH)

        #input_path_data = "../output_dataset_label/neural_network_data_small/"
        #input_neural_network = "../output_dataset_label/neural_network_output_TTT/"

       

        #output_path = "../output_dataset_label/TAGRETneural_network/"
        #output_neural_network = "../output_dataset_label/TAGRETneural_network_output/"
    
        evaluate_model(input_path_graphs, input_path_test_data, input_neural_network, output, EPOCH)
        print ("END_________________________________")
