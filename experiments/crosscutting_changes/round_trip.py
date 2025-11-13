import os

import random
import sys
import torch
from collections import defaultdict



current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)

sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)


from experiments.crosscutting_changes.helper.HELPER_final_eval import load_final_results
from experiments.crosscutting_changes.helper.HELPER_node_lables import compute_change_class, get_Predessors, mark_neightbors, remove_certain_info
from experiments.crosscutting_changes.helper.Completor import AllAtOnceCompletor, NextFocusCompletor
from experiments.crosscutting_changes.helper.HELPER_GENERIC import _graph_to_dict_general, save_to_json
from experiments.crosscutting_changes.helper.HELPER_eval import get_graphs
from experiments.crosscutting_changes.helper.HELPER_configuration import COMBINATION
from experiments.crosscutting_changes.helper.HELPER_eval import compute_precison_recall_perfile, load_test_data, save_and_print_statistics


device = torch.device('mps' if torch.has_mps else 'cuda' if torch.cuda.is_available() else 'cpu')

TRECHHOLD_UP= 1
TRECHHOLD_DOWN =0.5
K=10
name_sub_folders= "/default/"
completor_string = "All" #"NextFocus"

path_results = "../resultsmajorrevision/groundTruths/ours_test_17_10limit+json_promptadjusted"
whatToRemove = ['isPredessor', 'isSuccessor', 'hasChangedNeighbor', 'embedding']
#whatToNotQuess = ['isPredessor', 'isSuccessor', 'hasChangedNeighbor', 'embedding']
random.seed(42)
sampling_method = "ONE_STARTING_POINT_ONLY" #0.01 #"ONE_STARTING_POINT_ONLY"# "ONE_STARTING_POINT_ONLY" # 1, 0.1 -> needs to be int, percentage, 0.80 means a lot etc, 
# so number needs to be very low 

def choose_and_complete(graph, item_starting_node, max_completions, path_results, path_db, whatToRemove, test_file_map_graph, out_final_eval):
   
    #completor = AllAtOnceCompletor(path_results, path_db, whatToRemove)
    #completor.complete_model(graph, item_starting_node, max_completions, test_file_map_graph, out_final_eval)

    completor2 = NextFocusCompletor(path_results,path_db, whatToRemove)
    completor2.complete_model(graph, item_starting_node, max_completions, test_file_map_graph, out_final_eval)


def compute_gT_local_completion(graph, node_changed, to_remove, item_starting_node ): 

    # compute predessors 
    predecessors = defaultdict(list)
    graph_reduced_info = remove_certain_info(graph.copy(), to_remove)
    
    for n, _ in node_changed:
        #if n != str(item_starting_node): 
            # here _ = data which is the older version where nothing is removed yet 
            node_data = graph_reduced_info.nodes(data=True)[n]
            # remove all info the LLM does not have to guess correctly 
            #for k in to_remove:
            #   data.pop(k, None)

            preds = list(graph_reduced_info.predecessors(n))
            for p in preds: 
                #TODO get edhes between p and n and append too
                #CHANGED ↓ get edge data between p and n
                edge_data_list = []              

                edge_data_list.append(graph_reduced_info.get_edge_data(p, n))  #CHANGED

                 #CHANGED ↓ store structured info with edges
                predecessors[p].append({
                    "successor": n,
                    "node_data": node_data,
                    "edges": edge_data_list
                })  #CHANGED
            
    return predecessors
  
    # than lets remove the 

def devide_into_gT_alreadyExisting(graph, item_starting_node, path_results_filename): 

    # first of all we get the existing graph, + item_starting_node
    # so changed == false
    mark_neightbors(graph)
    changed, notchanged = compute_change_class(graph)
    notchanged_node_ids = [n[0] for n in notchanged]
    notchanged_node_ids.append(str(item_starting_node))
  
    existing_graph = graph.subgraph(notchanged_node_ids).copy()
    existing_graph.diff_id = graph.diff_id
   
    existing_graph = remove_certain_info(existing_graph, whatToRemove)
    save_to_json(path_results_filename, "existingGraph0", _graph_to_dict_general(existing_graph, existing_graph.diff_id))
    # die ersten node ids, also müssen wir das danach auch filtern, aber wo anders 

    # dann die ground truth für next focus node sind erstmal alle isPredssor = True
    # also vorhersagen der targets linearized_graph_data ist da result auch true 
    # außer halt eben den code den wir grad hinzugefügt haben
    groundTruth_nextFocus = get_Predessors(graph, item_starting_node)
    
    save_to_json(path_results_filename,"groundTruth_nextFocus0", groundTruth_nextFocus )

    # die gT für model completion ist dann aber die actual changed items, 
    # da wäre es aber gut zu wissen, jeweils noch den vorgänger zu haben, dass wir besser mappen können

    dict_localCompletion = compute_gT_local_completion(graph, changed, whatToRemove,item_starting_node ) 
    # here the stuff that does not have to be guessed should not be written down 
    save_to_json(path_results_filename,"groundTruth_localCompletion0", dict_localCompletion )


    print ("done with dataset creation and saving")

    return existing_graph,groundTruth_nextFocus, dict_localCompletion

def sample_candidates(candidates): 
    overall_number = len(candidates)
    if sampling_method == "ONE_STARTING_POINT_ONLY": 
        return [random.choice(candidates)[1]]

    else: 
        assert (0 < sampling_method <= 1) 
        sample_number = int (sampling_method * overall_number) 
        if sample_number < 1: 
            sample_number = 1

        # randomly sample without replacement, limited by candidate count
        sampled = random.sample(candidates, sample_number)

        # take the second element from each tuple
        return [item[1] for item in sampled]
        
        

        
    



def evaluate_model(input_path_graphs, input_path_test_data, path_db ,out_final_eval):
    
    out_final_eval = load_final_results(out_final_eval)
    out_final_eval = out_final_eval[out_final_eval["approach"] == "NeuralNetwork"]
    # Load test data
    all_test_node_pairs, all_test_labels = load_test_data(input_path_test_data)
   
    test_file_map = {
        file_path: (pairs, labels)
        for file_path, (pairs, labels) in zip(all_test_node_pairs.keys(), zip(all_test_node_pairs.values(), all_test_labels.values()))
    }

    allgraphs = get_graphs(input_path_graphs, name_sub_folders= "/default/")
    

    #We iterate over all graphs in test set
    for filename in test_file_map: 
        path_results_filename  = os.path.join(path_results, filename)
       
        # get the graph id 
        graph_id = test_file_map[filename][0][0][0] # first 0 fpr the actual input, second 0 for first element, third for graph id 
        # get the graph 
        graph = allgraphs[filename][graph_id] 
    
        # get a random beginning node which must be a changed node 
        # extract candidate node ids
        # probably we should do this with all 
        # out stuff vorkommen 
      
        candidates = [x for x, flag in zip(test_file_map[filename][0], test_file_map[filename][1]) if flag]
        random.seed(42)
        #  also get the maxium number of completions,  actual iterative prozedure tries
        max_completions = len(candidates) # if this is smaller we dont do ten iterations
        item_starting_node_list = sample_candidates(candidates)
        for item_starting_node in item_starting_node_list: 

            path_results_filename_subfolder = os.path.join(path_results_filename, f"starting_node_{str(item_starting_node)}")

            if os.path.exists(path_results_filename_subfolder):
                continue
            os.makedirs(path_results_filename_subfolder, exist_ok=True)

                
            # wäre gut wenn der candiat in preserved ist, dann müssen wir aber slicing von Tinnes anpassen
            # andernfalls müsste der nämlich aus GT entfernt werden , weil sonst bei tinnes immer der nextfocus correct ist
            # problem ist der ansatz geht nur auf wenn sich der value 1 auch geändert hat deshlab müssen wir das so mahcen 
            existing_graph, groundTruth_nextFocus, dict_localCompletion = devide_into_gT_alreadyExisting( graph,item_starting_node, path_results_filename_subfolder)
            choose_and_complete(existing_graph, item_starting_node, max_completions, path_results_filename_subfolder, path_db, whatToRemove, test_file_map[filename],  out_final_eval[out_final_eval["key"] == filename])


if __name__ == "__main__":
    
    #TODO needs to be changed 
    #input_path_graphs = "../output_dataset_label/dataset_node_embeddings_text-embedding-3-small-with-ids_small/diffgraphs/"
    input_path_graphs = "../output_dataset_label/dataset_node_embeddings_text-embedding-3-small-with-ids_majorRevision_onlytest_majorrevision_bucket3/diffgraphs/"
    #input_path_graphs = "../output_dataset_label/dataset_node_embeddings_text-embedding-3-small-with-ids/diffgraphs/"
    #input_path_graphs = "../output_dataset_label/dataset_node_embeddings_text-embedding-3-small-with-ids_majorRevision_onlytest_small/diffgraphs/"
   
   
    #TODO needs to be changed, this is the actual pairs, e.g train,t est desicription etc. 
    #input_path_test_data = "../output_dataset_label/embedding_data_refactored2"
    #input_path_test_data = "../resultsmajorrevision/datasets_traintestval_server"
    #input_path_test_data = "../resultsmajorrevision/datasets_majorrevision_server_good"
    input_path_test_data = "../resultsmajorrevision/datasets_majorrevision_bucket3"

    path_db = "../resultsmajorrevision/vector_database_new/"
    


    out_final_eval = "../Final_eval/output_ONLYONE4/" 
  


    evaluate_model(input_path_graphs, input_path_test_data, path_db, out_final_eval)
    print ("END_________________________________")
