import os
import pickle
import sys
import sys

import pandas as pd



current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Get the grandparent directory (parent of the parent)
grandparent_dir = os.path.dirname(parent_dir)

sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)
from experiments.crosscutting_changes.HELPER_GENERIC import sort_graphs_chrono, split_train_val_test
from modules.graph.io import load_components_networkx
from experiments.crosscutting_changes.HELPER_node_lables import compute_change_class, mark_neightbors
from experiments.crosscutting_changes.HELPER_configuration import WHAT_TO_CONSIDER

results = {}
print("python path")

name_sub_folders= "/default/" #"/diffgraphs/"
#input_path = "../output_dataset_label/dataset_node_embeddings_text-embedding-3-small-with-ids_small/diffgraphs/"
  
output_path ="../dataset_preprocessed/databases/"
name_db = "matrix_historial_per_graph_trainval.pkl"
input_path = "../output_dataset_label/dataset_node_embeddings_text-embedding-3-small-with-ids_small/diffgraphs/"


def build_matrix():
    matrix_per_graph={} #filename to matrix node x node , init with 0 
    for folder_name in os.listdir(input_path):
        
        if folder_name == 'modeling.mdt.uml2!!plugins_org.eclipse.uml2.uml_model_UML.ecore':
            print(f"Skipping {folder_name} (manually excluded)")
            continue


        # Skip files in the input_path
        if not os.path.isdir(input_path + '/' + folder_name):
            continue

        input_dir = input_path + folder_name + name_sub_folders
        graphs = load_components_networkx(data_folder=input_dir, mark_filename=True)
        graphs = sort_graphs_chrono(graphs)

       
        split_index_train ,_ = split_train_val_test(graphs)

        
        #we need to keep track of all nodes to extend the matrix correctly 
        allNodes=set()
        for i, graph in enumerate(graphs): 

            # Initialize as a list for the dataset if it doesn't exist
            # add the test graphs to the dataset and skip them 
            # fro constructing the matrix 

            #TODO continue for the test and val set, only add train
            if i >= split_index_train:
                continue

        
            #we collect all nodes that changed in this iteration
            #we also collect the predessors of these changed nodes, 
            #which are the nodes we would like to predict
            graph = mark_neightbors(graph)
            merged_changed_nodes, preserve_nodes = compute_change_class(graph, withEmbeddings=False)
            
            preserve_butneightbor  =  [  data["node_id"] for node, data in preserve_nodes if data.get(WHAT_TO_CONSIDER) is True]
            changed_butneightbor =  [ data["node_id"] for node, data in merged_changed_nodes if data.get(WHAT_TO_CONSIDER) is True]
            
            neightbors = preserve_butneightbor +changed_butneightbor

            #collect all the nodes and put them in the matrix 
            haschanged={data["node_id"] for node, data in merged_changed_nodes }
            preserve_nodes_id = {data["node_id"] for node, data in preserve_nodes }
            allNodes.update(haschanged)
            allNodes.update(preserve_nodes_id)

        
            # Initialize or extend the matrix size
            if folder_name not in matrix_per_graph:
                matrix_per_graph[folder_name] = pd.DataFrame(0, index=list(allNodes), columns=list(allNodes), dtype=int)
            else:
                existing_matrix = matrix_per_graph[folder_name]
                existing_nodes = set(existing_matrix.index)
                new_nodes = allNodes - existing_nodes

                if new_nodes:
                    # Create a DataFrame for new nodes
                    new_matrix = pd.DataFrame(0, index=list(new_nodes), columns=existing_matrix.columns)
                    
                    # Concatenate new rows
                    existing_matrix = pd.concat([existing_matrix, new_matrix], axis=0)

                    # Add new columns for the new nodes
                    new_columns = pd.DataFrame(0, index=existing_matrix.index, columns=list(new_nodes))
                    existing_matrix = pd.concat([existing_matrix, new_columns], axis=1)

                matrix_per_graph[folder_name] = existing_matrix

            # Update the matrix for all changed node pairs
            # the node that changed and all the predessors of changed nodes
            for node1 in haschanged:
                for node2 in neightbors:
                    #if node1 != node2:
                        matrix_per_graph[folder_name].loc[node1, node2] += 1

            print("done with dataset")
            print(folder_name)
    

    os.makedirs(os.path.dirname(output_path + name_db), exist_ok=True)
    with open(output_path + name_db, "wb") as f:
        pickle.dump(matrix_per_graph, f)


if __name__ == "__main__":
    build_matrix()

