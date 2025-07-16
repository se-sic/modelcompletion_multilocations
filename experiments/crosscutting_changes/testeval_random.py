import os

import pickle
import random
import sys



current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Get the grandparent directory (parent of the parent)
grandparent_dir = os.path.dirname(parent_dir)
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

# Add the grandparent directory to sys.path
sys.path.insert(0, grandparent_dir)
print(sys.path)
from experiments.crosscutting_changes.HELPER_configuration import WHAT_TO_CONSIDER
from experiments.crosscutting_changes.HELPER_eval import get_test_graphs

from experiments.crosscutting_changes.HELPER_eval import save_final_results
from experiments.crosscutting_changes.HELPER_node_lables import compute_change_class, mark_neightbors


import numpy as np
input_path_data = "../output_dataset_label/dataset_node_embeddings_text-embedding-3-small-with-ids_small/diffgraphs/"
output= "../results_final/"


name_sub_folders= "/default/" #"/diffgraphs/"


input_path_splitting = "../output_dataset_label/embedding_data_refactored2"

TopK= 5   

def get_graph_nodes(graph): 
    

    # Compute class for the graph
    graph = mark_neightbors(graph)
    merged_changed_nodes, preserve_nodes = compute_change_class(graph, withEmbeddings=False)

    all_nodes= merged_changed_nodes + preserve_nodes

    #these are the ground truth values
    preserve_butneightbor  =  [data["node_id"] for node, data in preserve_nodes if data.get(WHAT_TO_CONSIDER) is True]
    changed_butneightbor =  [data["node_id"] for node, data in merged_changed_nodes if data.get(WHAT_TO_CONSIDER) is True]
    neightbors = preserve_butneightbor +changed_butneightbor

    merged_changed_nodes_ids =  {int(node) for node, data in merged_changed_nodes  }

    return all_nodes,merged_changed_nodes_ids, neightbors


def eval_random():
    results={}
    
   
    test_graphs=get_test_graphs(input_path_data, input_path_splitting,  name_sub_folders)

    for folder_name, graphs in test_graphs.items():

        if folder_name == 'modeling.mdt.uml2!!plugins_org.eclipse.uml2.uml_model_UML.ecore':
            print(f"Skipping {folder_name} (manually excluded)")
            continue
         
        total_true_positives = 0
        total_predictions = 0

        precision_list = []
        meta_datas=[]
        labels_graph=[]
        all_probabilities=[]

        for graph in graphs:

            graph_id = int(graph.diff_id.split("_")[1].split(".")[0])
            all_nodes, merged_changed_nodes_ids, neightbors = get_graph_nodes(graph)
            name_to_id = {int(id) : data["node_id"] for id, data in all_nodes}
            # Predict top 5 elements most likely to change with each changed node
            for node_id in merged_changed_nodes_ids:

                max_predictions = min (TopK, len(neightbors) )
                total_predictions += max_predictions

                # keep the original order of `all_nodes`
                ordered_nodes = [int(id) for id, data in all_nodes]

                # meta information for every candidate of THIS source-node
                meta_data = [
                    (graph_id, node_id, cand_id, -1)              # (graph, src, tgt, dummy)
                    for cand_id in ordered_nodes
                ]

                # uniform “probabilities” – random baseline has no real scores
                probabilities = [1.0 / len(ordered_nodes)] * len(ordered_nodes)

                # binary relevance labels
                labels = [1 if name_to_id[cand_id] in neightbors else 0 for cand_id in ordered_nodes]

                # store for possible downstream analysis
                meta_datas.append(meta_data)
                labels_graph.append(labels)
                all_probabilities.append(probabilities)
                # Calculate precision metrics
                top_predictions = random.sample(ordered_nodes, max_predictions) 
                true_positives = sum(1 for pred in top_predictions if name_to_id[pred] in neightbors)
                total_true_positives += true_positives

                precision_k = true_positives/max_predictions
                precision_list.append(precision_k)
              
              

        #we compute the average per file 
        avg_precision = np.mean(precision_list) if precision_list else -1

        #IMPORTANT: per example (one node) we take the precision_k and than average it again 
        result= {
            "precision_avg": avg_precision,
            "precision_all": precision_list,
            "total_true_positives": total_true_positives,
            "total_predictions": total_predictions, 
            "meta": meta_datas, 
            "labels": labels_graph, 
            "probabilities": all_probabilities
        }

        results[folder_name]= result

        print(f"folder_name {folder_name}")
        print(f"Overall Precision: {avg_precision:.4f} ({total_true_positives}/{total_predictions})")
        print(f"overall number graphs: {len(graphs)}")
        print(f"number focus nodes:{len(merged_changed_nodes_ids)}")
        os.makedirs(output, exist_ok=True)

    save_final_results(output, results, "randombaseline")

if __name__ == "__main__":
    eval_random()