import ast
import json
import math
import os
import sys
import sys



print("python path")
current_dir = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Get the grandparent directory (parent of the parent)
grandparent_dir = os.path.dirname(parent_dir)
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

# Add the grandparent directory to sys.path
sys.path.insert(0, grandparent_dir)
print(sys.path)
from modules.graph.io import load_components_networkx, save_components_networkx
from scripts.compute_connected_components import assign_ids_to_diff_graphs
from experiments.crosscutting_changes.HELPER_node_emb import get_node_embeddings, individual_embeddings
from modules.domain_specific.change_graph import clean_up_string, clean_up_string_ast_literal




CONSIDERED_CONTEXT_NODE ="EASY"

#only if choosing not easy you need to select these
#FOCUS_CURRENT = "ALL" #"ALL" #"SUC_PREC", "SUC", "PREC"
#FOCUS_FUTURE = "SUC_PREC"

input_path ="../dataset_raw/diffgraph_new_all/"
name_sub_folders= "/default/" #"/diffgraphs/"
output_path = "../output_dataset_label/dataset_node_embeddings_text-embedding-3-small-with-ids-indivual-embeddings/diffgraphs/"


skipped_nodes=0
number_nodes =0
EASY_EMBEDDING =False

# Loop over all datasets 
for folder_name in os.listdir(input_path):

    list_diffgraph_id_to_cc = {}
    # Skip files in the input_path
    if not os.path.isdir(input_path + '/' + folder_name):
        continue

    full_folder_path = os.path.join(output_path, folder_name)
    if os.path.exists(full_folder_path):
        continue

    nb_diffs, nb_eos, pertubation = ("None", "None", "None")

    # Generate name for the output folder
    input_dir = input_path + folder_name + name_sub_folders
    output_dir = output_path + folder_name + name_sub_folders
    graphs = load_components_networkx(data_folder=input_dir, mark_filename=True)
    #assign_ids_to_diff_graphs(graphs, folder_name, is_already_diffed=True)


    for graph in graphs: 
        all_nodes = []
        preserve_nodes = set()
        deleted_nodes = set()
        added_nodes = set()
        changed_nodes = set()
        flagged_as_neighbor_of_changed =set()


        # Iterate through the nodes and add them to the list
        for node, data in graph.nodes(data=True):
            number_nodes+=1
            # Extract the 'changeType' value from the 'label' attribute
            try: #node_data = eval(data['label'])
               # node_data = json.loads(data['label'])
                node_data = ast.literal_eval(clean_up_string_ast_literal(data['label']))
                node_id=node_data['attributes']['id']
                attributes = node_data['attributes']
                type_node = node_data['type']
                class_name =  node_data['className']
            
    



            except (ValueError, SyntaxError) as e:
                print(f"Error evaluating node data for graph {graph.name}. Skipping node.")
                skipped_nodes+=1 #does not cont
                # if this hapens it is usally the whole graph, so we reset 
                all_nodes = []
                break  # Skip the rest of the loop for this iteration
            #removes the elemnt to compute the embeeding
            # THE MOST IMPORTANT LINE
            change_type = node_data.pop('changeType', None)
            
            # Categorize the nodes based on the 'changeType'
            if change_type == 'Preserve':
                preserve_nodes.add(node)
            elif change_type == 'Deleted' or change_type == 'Remove' or change_type == 'Delete':
                deleted_nodes.add(node)
            elif change_type == 'Change': 
                changed_nodes.add(node)
            elif change_type == 'Add':
                added_nodes.add(node)
            else:
                # Raise an error if the changeType is none of the expected values
                raise ValueError(f"Unexpected changeType: {change_type} for node {node}")
          
            all_nodes.append(
                (
                    node,
                    {
                        'label': str(node_data),
                        'graph': graph,
                        'attributes': str(attributes),
                        'node_id': node_id,  # or however you're deriving this id
                        'type': type_node,           # or however you're getting this type
                        'class_name': class_name     # or however you're getting this class name
                    }
                )
            )
           
        #important that this is all_nodes, not graph.nodes, because parsing
        if (len(all_nodes) < 1): 
            #not able to parse or compute embeedings from graph, or it was empty, so we ignore
            print("ignoring graph")
            continue 
        
        #now lets compute the embeddings for the nodes 
        if CONSIDERED_CONTEXT_NODE=="EASY": 
            print(folder_name)
            print(str(graph.diff_id))
            if EASY_EMBEDDING: 
                node_embeddings = get_node_embeddings(all_nodes)
            else: 
                node_embeddings = individual_embeddings(all_nodes)

           

        #add the embeddings to the graph
        for node in all_nodes: 
            id = node[0]
            
            
            graph.nodes[id]["embedding"] = node_embeddings[id]
        save_components_networkx(graph, output_dir, graph.diff_id)
print ("skipped_nodes")
print (skipped_nodes)
print ("number of all nodes")
print(number_nodes)
    
    
        

            


    


 
