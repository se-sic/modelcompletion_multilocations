import os
import sys
import sys


print("python path")
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Get the grandparent directory (parent of the parent)
grandparent_dir = os.path.dirname(parent_dir)

sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)

from experiments.crosscutting_changes.HELPER_node_lables import compute_change_class
from modules.graph.io import load_components_networkx
from experiments.crosscutting_changes.HELPER_similarity import compute_similarity_between_nodes

AVERAGE = False
WHICH_NODES=  "EASY" #"NEIGHTBORS"              

INCLUDE_NEIGHBORS = False #include the local stuff in thr general statisitcs. local stuff is more likely to be similar
input_path ="../output_dataset_label/dataset_node_embeddings_text-embedding-3-small/diffgraphs/"
name_sub_folders= "/default/" #"/diffgraphs/"
output_path = "../output_dataset_label/statistics/newembedding/"

COMPUTE_SIMIL=True
graph_id=0; 
overall_number_changed_preserved={}
overall_similarity_results = []
skipped_nodes=0


def execute():
    os.makedirs(output_path, exist_ok=True)
    # Loop over all datasets
    for folder_name in os.listdir(input_path):


        # Skip files in the input_path
        if not os.path.isdir(input_path + '/' + folder_name):
            continue

        # Generate name for the output folder
        input_dir = input_path + '/' + folder_name + name_sub_folders
        output_dir = output_path + '/' + folder_name

        graphs = load_components_networkx(data_folder=input_dir)
        #assign_ids_to_diff_graphs(graphs, folder_name)
        for graph in graphs: 

            merged_changed_nodes, preserve_nodes = compute_change_class(graph)
            
            if (len(merged_changed_nodes) != 2 and len(preserve_nodes)<= len(merged_changed_nodes)): 
                #print(f"Graph {graph.name} has {len(merged_changed_nodes)} changed nodes. Skipping graph.")
                continue

            overall_number_changed_preserved[graph_id]= {
                    'num_changed': len(merged_changed_nodes),
                    'num_preserved': len(preserve_nodes) }
            graph_id+=1

            
            if (merged_changed_nodes and preserve_nodes and COMPUTE_SIMIL):

                
                # define which graph embeddings should be used fro comparison
                # just compare the changed nodes and preserved nodes
                if (WHICH_NODES== "EASY"): #"NEIGHTBORS"
                    preserve_node_embeddings = {node: data["embedding"] for node, data in preserve_nodes}
                    merged_node_embeddings =  {node: data["embedding"] for node, data in merged_changed_nodes}
                    merged_node_embeddings_comparing =  {node: data["embedding"] for node, data in merged_changed_nodes}
                elif (WHICH_NODES==  "NEIGHTBORS"): 
                    preserve_node_embeddings = {node: data["embedding"] for node, data in preserve_nodes if data["hasChangedNeighbor"] is False}
                    
                    merged_node_embeddings =  {node: data["embedding"] for node, data in merged_changed_nodes}

                    preserve_butneightbor  =  {node: data["embedding"] for node, data in preserve_nodes if data.get("hasChangedNeighbor") is True}
                    changed_butneightbor =  {node: data["embedding"] for node, data in merged_changed_nodes if data.get("hasChangedNeighbor") is True}
                    merged_node_embeddings_comparing =  preserve_butneightbor | changed_butneightbor

                    #if these are empty we dont have to look at this graph ansymore
                    if not preserve_node_embeddings or not merged_node_embeddings_comparing: 
                        print ("len(preserve_node_embeddings)")
                        print (len(preserve_node_embeddings))
                        print ("len(merged_node_embeddings_comparing)")
                        print (len(merged_node_embeddings_comparing))
                        print("no nodes")
                        continue; 
                


                similarity_results_nodes_in_graph = {}
            
            
                for node, data in merged_changed_nodes:
                    # Compute average similarity with all other preserved nodes
                    changed_node_embedding = merged_node_embeddings[node]
                    
                    similarity_preserved , average_preserved = compute_similarity_between_nodes(changed_node_embedding, preserve_node_embeddings)

                
                    # define which nodes we exclude from the comparison
                    graph = data['graph']
                    neightbors= list(graph.neighbors(node))
                    if (INCLUDE_NEIGHBORS):
                        exclude_nodes = [node] + neightbors
                    else: 
                        exclude_nodes = [node]

                    
                    # Compute average similarity with all other changed nodes
                    other_merged_changed_nodes = {k: v for k, v in merged_node_embeddings_comparing.items() if k not in exclude_nodes}
                
                    if other_merged_changed_nodes:
                        similarity_merged_changed , average_changed= compute_similarity_between_nodes(changed_node_embedding, other_merged_changed_nodes)
                    else: 
                        print("No further away changed items")
                        continue

                    if node in similarity_results_nodes_in_graph:
                        raise ValueError(f"Attempted to overwrite existing node data for node: {node}")
                    else:
                        if AVERAGE: 
                            similarity_results_nodes_in_graph[node] = {
                                'similarity_to_preserved': average_preserved,
                                'similarity_to_merged_changed': average_changed }
                        else:
                            similarity_results_nodes_in_graph[node] = {
                            'similarity_to_preserved': similarity_preserved,
                            'similarity_to_merged_changed': similarity_merged_changed }

                overall_similarity_results.append(similarity_results_nodes_in_graph)



    ########################################
    ###
    ###displaying stuff
    ########################################        
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import ttest_rel

    #print the number of graphs we are considering
    print("Skipped nodes:", skipped_nodes)
    print("Number of graphs considered:", len(overall_number_changed_preserved))
    # Assuming overall_similarity_results is a list of dictionaries with the similarity data
    if(COMPUTE_SIMIL):

        differences = []
        preserved_similarities = []
        merged_changed_similarities = []
        # Collect differences
        for graph_result in overall_similarity_results:
            if AVERAGE: 
                for node, similarities in graph_result.items():
                    preserved_similarities.append(similarities['similarity_to_preserved'])
                    merged_changed_similarities.append(similarities['similarity_to_merged_changed'])
            else: 
                for node, similarities in graph_result.items():
                    for similarity in np.nditer(similarities['similarity_to_preserved']):
                        preserved_similarities.append(similarity.item())  # .item() to convert numpy types to Python scalars
                
                    for similarity in np.nditer(similarities['similarity_to_merged_changed']):
                        merged_changed_similarities.append(similarity.item())  # .item() for type consistency


        # Convert to numpy arrays for statistical analysis
        preserved_similarities = np.array(preserved_similarities)
        merged_changed_similarities = np.array(merged_changed_similarities)


        # Plot the two distributions
        plt.figure(figsize=(10, 6))
        plt.hist(preserved_similarities, bins=30, alpha=0.7, label='Preserved Similarities', color='blue', edgecolor='k')
        plt.hist(merged_changed_similarities, bins=30, alpha=0.7, label='Merged Changed Similarities', color='red', edgecolor='k')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.title('Distribution of Cosine Similarities')
        plt.legend()
        plt.savefig(output_path+'plot_cosine_similarity_new.png')  # Saves the plot as a PNG file
        plt.close()
        print("saved png")


    # Extracting data into lists
    num_changed = [info['num_changed'] for info in overall_number_changed_preserved.values()]
    num_preserved = [info['num_preserved'] for info in overall_number_changed_preserved.values()]

    # Convert lists to numpy arrays for plotting
    num_changed_array = np.array(num_changed)
    num_preserved_array = np.array(num_preserved)

    difference_array = num_preserved_array - num_changed_array

    # Create the figure and plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(difference_array, bins=100,range=(-100, 50), alpha=0.7, color='green', edgecolor='k')
    plt.xlabel('Difference in Count (Preserved - Changed)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Differences Between Preserved and Changed Nodes Across Graphs')
    plt.savefig(output_path+'histogram_plot_differences_new.png')  # Saves the plot as a PNG file
    plt.close()



#execute()