#!/usr/bin/python
import math
import time
import os, sys
from modules.graph.graph_operations import split_in_connected_components, filter_too_large
from modules.graph.io import load_components_networkx
from modules.textual.io import export_TLV



def assign_ids_to_diff_graphs(graphs, folder_name, is_already_diffed=False):
    index = 0
    for graph in graphs:
        if is_already_diffed:
            # Assumes the file name is in the format: diff_<number>.json
            diff_num_str = graph.file_name.split('_')[1].split('.')[0]
            graph.diff_id = int(diff_num_str)
        else:
            graph.diff_id = index
            index += 1
        graph.folder_name = folder_name


#TODO this is a future-jupyter notebook
def main(input_path, output_path, max_number_nodes_cc, max_number_edges_cc):
    dict_folder_name_to_diff_graphs_to_cc = {}
    # Create output folder
    os.makedirs(output_path, exist_ok=True)

    # path for the output of the csv file
    results_path = output_path + '/results.csv'
    if os.path.exists(results_path):
        print(f"WARN: There was already a results file in {output_path}.")
        os.remove(results_path)

    # Write the header of the results file
    with open(results_path, 'w') as f:
        f.write(
            "Id;Diffs;EOs;Pertubation;Components;Nodes;Edges;Filtered;Component_Computation_Time;Filter_Correct;Correct_1_Matches;Correct_2_Matches;Correct_3_Matches\n")

    # Loop over all datasets
    for folder_name in os.listdir(input_path):

        list_diffgraph_id_to_cc = {}
        # Skip files in the input_path
        if not os.path.isdir(input_path + '/' + folder_name):
            continue

        nb_diffs, nb_eos, pertubation = ("None", "None", "None")

        # Generate name for the output folder
        input_dir = input_path + '/' + folder_name + '/diffgraphs/'
        output_dir = output_path + '/' + folder_name

        # Compute connected components
        start_time = time.time()

        graphs = load_components_networkx(data_folder=input_dir)

        assign_ids_to_diff_graphs(graphs, folder_name)

        components, nb_of_components_per_diff, id_diffgraphs_per_compo = split_in_connected_components(graphs)
        components,id_diffgraphs_per_compo, filtered = filter_too_large(components,id_diffgraphs_per_compo, filtered={}, nb_nodes=max_number_nodes_cc, nb_edges=max_number_edges_cc)

        for comp in components:
            if comp.diff_id not in list_diffgraph_id_to_cc:
                list_diffgraph_id_to_cc[comp.diff_id] = []
            list_diffgraph_id_to_cc[comp.diff_id].append(comp)

        dict_folder_name_to_diff_graphs_to_cc[folder_name]= list_diffgraph_id_to_cc

        end_time = time.time()
        computation_time = str(end_time - start_time)

        # Count number of nodes
        nb_nodes = sum([len(component.nodes()) for component in components])
        nb_edges = sum([len(component.edges()) for component in components])


        # Exports
        os.makedirs(output_dir, exist_ok=True)
        export_TLV(components,id_diffgraphs_per_compo,  output_dir + '/connected_components.lg')

        # Write csv
        with open(results_path, 'a') as f:
            f.write(
                f"{folder_name};{nb_diffs};{nb_eos};{pertubation};{len(components)};{nb_nodes};{nb_edges};{filtered};{computation_time};False;"";"";""\n")


    return dict_folder_name_to_diff_graphs_to_cc

if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2], max_number_nodes_cc=math.inf, max_number_edges_cc= math.inf)

    elif len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print(
            "Unexpected number of arguments. At least input path")
