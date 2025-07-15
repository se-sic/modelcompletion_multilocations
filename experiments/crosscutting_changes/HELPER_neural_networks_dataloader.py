import sys
import os



current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Get the grandparent directory (parent of the parent)
grandparent_dir = os.path.dirname(parent_dir)

sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)

import random

import torch
from experiments.crosscutting_changes.HELPER_configuration import ADJUST_TRAIN_TEST_VAL_RATIO, BALANCE_METHOD, COMBINATION, ADJUST_TRAIN_SIZE, MAKE_FOLDERS_EQUAL_SIZE_OR_CUT_ABOVE, REMOVE_DUPLICATES, TRAIN_RATIO, TRAIN_TEST_SPLIT, TRAIN_TEST_VAL_RATIO, VAL_RATIO , WHAT_TO_CONSIDER

from experiments.crosscutting_changes.HELPER_GENERIC import print_distribtutions, print_split_counts, set_random_seed
from torch.utils.data import Subset, TensorDataset
from typing import Counter
from experiments.crosscutting_changes.HELPER_dataset_filtering import change_trainset_size, ensure_train_val_test_ratio, remove_duplicates


import os


import os
import sys

import numpy as np




current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Get the grandparent directory (parent of the parent)
grandparent_dir = os.path.dirname(parent_dir)

sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)

from experiments.crosscutting_changes.HELPER_GENERIC import append_to_pth
from experiments.crosscutting_changes.HELPER_GENERIC import sort_graphs_chrono
from modules.graph.io import load_components_networkx
from experiments.crosscutting_changes.HELPER_node_lables import compute_change_class, mark_neightbors


name_sub_folders= "/default/"


def load_train_test_val_data( input_path ):
    all_data={}

    subfolders = [subfolder for subfolder in os.listdir(input_path)
                  if os.path.isdir(os.path.join(input_path, subfolder))]


    for folder_name in subfolders:
        print(folder_name)
        subfolder_path = os.path.join(input_path, folder_name)

        if folder_name not in all_data:
            all_data[folder_name] = {}
        
        for split in ["train", "val", "test"]:
                if split not in all_data[folder_name]:
                    all_data[folder_name][split] = {"pairs": [], "labels": []}

        # Ensure it's a directory
        if os.path.isdir(subfolder_path):
            # Paths to train and test files
            train_file = os.path.join(subfolder_path, "dataset_pairs_train.pth")
            test_file = os.path.join(subfolder_path, "dataset_pairs_test.pth")
            val_file =  os.path.join(subfolder_path, "dataset_pairs_val.pth")

            # Load train data if available
            if os.path.exists(train_file):
                print(f"Loading: train data for folder: {folder_name}")
                print(f"Size of {train_file}: {os.path.getsize(train_file) / (1024**2):.2f} MB")
                
                train_data = torch.load(train_file)
                train_pairs = train_data["pairs"]
                train_labels = train_data["labels"]
                target="train"

                all_data[folder_name][target]["pairs"]= train_pairs
                all_data[folder_name][target]["labels"]= train_labels

                 # Count stats
                total = len(train_labels)
                true_count = sum(train_labels)
                print(f"{folder_name} - {target}: total={total}, label=True={true_count}")




            if (os.path.exists(val_file)): 
                print(f"Loading: val data for folder: {folder_name}")
                print(f"Size of {val_file}: {os.path.getsize(val_file) / (1024**2):.2f} MB")
                val_data = torch.load(val_file)
                val_pairs = val_data["pairs"]
                val_labels = val_data["labels"]
                target="val"
               
                all_data[folder_name][target]["pairs"]= val_pairs
                all_data[folder_name][target]["labels"]= val_labels



            # Load test data if available
            if os.path.exists(test_file):
                print(f"Loading: test data for folder: {folder_name}")
                print(f"Size of {test_file}: {os.path.getsize(test_file) / (1024**2):.2f} MB")
                test_data = torch.load(test_file)
                test_pairs = test_data["pairs"]
                test_labels = test_data["labels"]
                target="test"
                
                all_data[folder_name][target]["pairs"]= test_pairs
                all_data[folder_name][target]["labels"]= test_labels
       

    return all_data



# make pairs, and split each subfolder into train, test, val 
def preprocess_data_construct_pairs(input_path, output_dir): 

    all_data={}

    # Skip processing if the files already exist
    if os.path.exists(output_dir):
        
        print(f"Files already exist for {output_dir}. Skipping...")
        all_data= load_train_test_val_data(output_dir)

    if (False): 


        for folder_name in os.listdir(input_path):

            file_path_train = os.path.join(output_dir, folder_name, "dataset_pairs_train.pth")
            file_path_test = os.path.join(output_dir, folder_name, "dataset_pairs_test.pth")
            file_path_val =  os.path.join(output_dir, folder_name, "dataset_pairs_val.pth")

            if all(os.path.exists(p) for p in [file_path_train, file_path_val, file_path_test]):
                continue


            # Skip files in the input_path
            if not os.path.isdir(input_path + '/' + folder_name):
                continue

            # Generate name for the output folder
            input_dir = input_path + '/' + folder_name + name_sub_folders
            
            # do the dataset splitting
            graphs = load_components_networkx(data_folder=input_dir, mark_filename=True)
            graphs = sort_graphs_chrono(graphs)

            ### WHAT TO CONSIDER TEST; TRAIN ; VAL 
            if (TRAIN_TEST_VAL_RATIO == "ONE_ONLY"): 
              
                # Ensure at least 2 graphs to extract 1 val and 1 test
                if len(graphs) < 3:
                    print(f"Skipping {folder_name} — not enough graphs for ONE_ONLY split")
                    continue

                split_index_train = len(graphs) - 2
                split_index_val = len(graphs) - 1
               

            #oder 90,10,10
            elif TRAIN_TEST_VAL_RATIO == "PERCENTAGES" : 
                split_index_train = int(len(graphs) * TRAIN_RATIO) 
                split_index_val = split_index_train + int(len(graphs) * VAL_RATIO)

            for i, graph in enumerate(graphs): 

                graph_id = int(graph.diff_id.split("_")[1].split(".")[0])
                
            
                graph = mark_neightbors(graph)
                merged_changed_nodes, preserve_nodes = compute_change_class(graph)

                preserve_noneightbor= {node: data["embedding"] for node, data in preserve_nodes if data[WHAT_TO_CONSIDER] is False}
                preserve_butneightbor  =  {node: data["embedding"] for node, data in preserve_nodes if data.get(WHAT_TO_CONSIDER) is True}

                changed_noneightbor = {node: data["embedding"] for node, data in merged_changed_nodes if data.get(WHAT_TO_CONSIDER) is False}
                changed_butneightbor =  {node: data["embedding"] for node, data in merged_changed_nodes if data.get(WHAT_TO_CONSIDER) is True}
            

                neightbor_not_changed = preserve_noneightbor | changed_noneightbor
                neightbor_changed = preserve_butneightbor | changed_butneightbor

                merged_node_embeddings =  {node: data["embedding"] for node, data in merged_changed_nodes}

                
                #TODO a small adjustment here could be to remove the self mapping so chek item != item 
                # True pairs: combine all nodes in merged_changed_nodes with themselves
                true_pairs = [(graph_id, int(n1), e1, int(n2), e2) for n1,e1 in merged_node_embeddings.items() for n2,e2 in neightbor_changed.items()] #if n1 != n2]
                true_labels =[True] * len(true_pairs) 

                # False pairs: combine all nodes in merged_changed_nodes with preserve_nodes
                false_pairs=[(graph_id, int(n1), e1, int(n2), e2) for n1,e1 in merged_node_embeddings.items() for n2,e2 in neightbor_not_changed.items()]# if n1 != n2]
                false_labels=[False] * len(false_pairs)

                # Combine True and False pairs
                combined_pairs = true_pairs + false_pairs
                combined_labels = true_labels + false_labels

                # Assign the target file based on the split
                if i < split_index_train:
                    target = "train" 
                    target_file = file_path_train  # Training set
                elif i < split_index_val:
                    target = "val" 
                    target_file = file_path_val  # Validation set
                else:
                    target = "test"
                    target_file = file_path_test 


                # Append data to the appropriate file
                if folder_name not in all_data:
                    all_data[folder_name] = {}

                for split in ["train", "val", "test"]:
                    if split not in all_data[folder_name]:
                        all_data[folder_name][split] = {"pairs": [], "labels": []}

                all_data[folder_name][target]["pairs"]= combined_pairs
                all_data[folder_name][target]["labels"]= combined_labels

                

                os.makedirs(os.path.dirname(target_file), exist_ok=True)
                append_to_pth(target_file, combined_pairs, combined_labels)

    return all_data
            



### filter the data ########################
def filter_dataset(all_data, sample ):

    all_train_node_pairs = []
    all_train_labels = []
    all_test_node_pairs = []
    all_test_labels = []
    all_val_node_pairs = []
    all_val_labels = []

    subfolders = list(all_data)
    selected_subfolders = random.sample(subfolders, min(sample, len(subfolders)))

    print(f"Selected subfolders: {selected_subfolders}")
    
    print ("\n DATASET BEFORE FILTERTING")
    print_split_counts(all_data)


    for subfolder in selected_subfolders:

        train_data = all_data[subfolder]["train"]
        val_data   = all_data[subfolder]["val"]
        test_data  = all_data[subfolder]["test"]

        train_pairs = train_data["pairs"]
        train_labels = train_data["labels"]
       
        val_pairs = val_data["pairs"]
        val_labels = val_data["labels"]
       
        test_pairs = test_data["pairs"]
        test_labels = test_data["labels"]
        
        ######preprocessing of our test-train-val split ############
        # wir wollen nur train und test datenset 
        if (TRAIN_TEST_SPLIT == "TRAINTEST"):
                
            train_pairs = train_pairs + val_pairs
            train_labels = train_labels + val_labels
            #If you want to force val_pairs and val_labels to be empty (e.g. to skip validation and do training only), you can do this safely in PyTorch with:
            val_labels=[]
            val_pairs=[]


        ###### preprocessing of our training data ######
        if ADJUST_TRAIN_SIZE!= "DO_NOT_CUT":

            # --- Balancieren auf target_samples_per_folder ---
            train_pairs, train_labels = change_trainset_size(
                train_pairs,
                train_labels,
                ADJUST_TRAIN_SIZE,
                MAKE_FOLDERS_EQUAL_SIZE_OR_CUT_ABOVE)

        ###### finally append  #######

        all_train_node_pairs.append(train_pairs)
        all_train_labels.append(train_labels)

        all_val_node_pairs.append(val_pairs)
        all_val_labels.append(val_labels)

        all_test_node_pairs.append(test_pairs)
        all_test_labels.append(test_labels)


    ###### remove duplicates ######

    if REMOVE_DUPLICATES:
        assert COMBINATION != "ALL_EMBEDDINGS"
        #please make assert than muss auch COMBINATION == "ALL_EMBEDDINGS
        all_train_node_pairs, all_train_labels= remove_duplicates(all_train_node_pairs, all_train_labels)
    

        ###### preprocessing of our train-test-data split ratio  ########

    if ADJUST_TRAIN_TEST_VAL_RATIO:
        ensure_train_val_test_ratio(all_train_node_pairs, all_train_labels,
                                all_val_labels, all_test_labels) 


    print ("\n DATASET AFTER FILTERTING")
    print_distribtutions(all_train_labels, all_test_labels, all_val_labels)

        
    #print_distribtutions(all_train_labels, all_test_labels, all_val_labels)
    return all_train_node_pairs, all_train_labels, all_test_node_pairs, all_test_labels, all_val_node_pairs, all_val_labels

def convert_vector_pairs_only(nested_node_pairs, dtype=torch.float32):
    """
    Converts a nested list of 5-tuples to a (N, 2, D) tensor.
    Each element of the input must be a list of tuples like:
        (id1, id2, vec1, id3, vec2)
    """
    if not nested_node_pairs:
        return torch.empty((0, 2, 0), dtype=dtype)

    vec_pairs = [
        [vec1, vec2]
        for sublist in nested_node_pairs
        for (_, _, vec1, _, vec2) in sublist
    ]
    return torch.tensor(vec_pairs, dtype=dtype)

def convert_embeddings_all(nested_node_pairs, dtype=torch.float32):
    """
    Converts a nested list of 5-tuples to a (N, 2, K, D) tensor,
    where each vector is a dict of embedding types (keys).
    """
    if not nested_node_pairs:
        return torch.empty((0, 2, 0, 0), dtype=dtype)

    vec_pairs = []
    for sublist in nested_node_pairs:
        for (_, _, vec1, _, vec2) in sublist:
            # Get consistent order of keys
            keys = sorted(vec1.keys())
            vec1_parts = [vec1[k] for k in keys]
            vec2_parts = [vec2[k] for k in keys]
            vec_pairs.append([vec1_parts, vec2_parts])  # shape (2, K, D)

    return torch.tensor(vec_pairs, dtype=dtype)  # final shape (N, 2, K, D)


def convert_vectors_and_ids(nested_node_pairs, dtype=torch.float32):
    """
    Converts a nested list of 5-tuples to a (N, 2, D) tensor.
    Each element of the input must be a list of tuples like:
        (id1, id2, vec1, id3, vec2)
    """
    if not nested_node_pairs:
        return torch.empty((0, 2, 0), dtype=dtype)

    vec_pairs = [
        [gId, node1id, vec1, node2id, vec2]
        for sublist in nested_node_pairs
        for (gId, node1id, vec1, node2id, vec2) in sublist
    ]
    return torch.tensor(vec_pairs, dtype=dtype)



def create_final_dataset(all_train_node_pairs, all_train_labels, all_test_node_pairs,all_test_labels, all_val_node_pairs, all_val_labels):
    
   

    if (COMBINATION == "VECTOR_ONLY"): 
        if all_train_node_pairs:
            all_train_node_pairs = convert_vector_pairs_only(all_train_node_pairs)

        if all_test_node_pairs:
            all_test_node_pairs = convert_vector_pairs_only(all_test_node_pairs)

        if all_val_node_pairs:
            all_val_node_pairs = convert_vector_pairs_only(all_val_node_pairs)
        else:
            all_val_node_pairs = torch.empty((0, 2, 0), dtype=torch.float32)
            all_val_labels = torch.empty((0,), dtype=torch.float32)
        

        
    elif (COMBINATION == "VECTOR_AND_IDS"): 
        if all_train_node_pairs:
            all_train_node_pairs = convert_vectors_and_ids(all_train_node_pairs)

        if all_test_node_pairs:
            all_test_node_pairs = convert_vectors_and_ids(all_test_node_pairs)

        if all_val_node_pairs:
            all_val_node_pairs = convert_vectors_and_ids(all_val_node_pairs)
        else:
            all_val_node_pairs = torch.tensor([], dtype=torch.float32)
            all_val_labels= torch.tensor([], dtype=torch.float32) 

        
    elif (COMBINATION == "ALL_EMBEDDINGS"): 
        if all_train_node_pairs:
            all_train_node_pairs = convert_embeddings_all(all_train_node_pairs)

        if all_test_node_pairs:
            all_test_node_pairs = convert_embeddings_all(all_test_node_pairs)

        if all_val_node_pairs:
            all_val_node_pairs = convert_embeddings_all(all_val_node_pairs)
        else:
            all_val_node_pairs = torch.tensor([], dtype=torch.float32)
            all_val_labels= torch.tensor([], dtype=torch.float32) 
        

    # Flatten before conversion
    all_train_labels = [item for sublist in all_train_labels for item in sublist]
    all_test_labels  = [item for sublist in all_test_labels  for item in sublist]
    all_val_labels   = [item for sublist in all_val_labels   for item in sublist]

    # Convert to float tensors
    all_train_labels = torch.tensor(all_train_labels, dtype=torch.float32)
    all_test_labels  = torch.tensor(all_test_labels, dtype=torch.float32)
    all_val_labels   = torch.tensor(all_val_labels, dtype=torch.float32)

    all_train_labels = torch.tensor(all_train_labels, dtype=torch.float32)
    all_test_labels = torch.tensor(all_test_labels, dtype=torch.float32)
    all_val_labels = torch.tensor(all_val_labels, dtype=torch.float32)


    ############################################################
    ### NOW LETS BALANCE THE DATA ##############################
    ############################################################
    X_train, y_train = all_train_node_pairs, all_train_labels
    X_test, y_test = all_test_node_pairs, all_test_labels
    X_val, y_val = all_val_node_pairs, all_val_labels

    # Create DataLoaders for batching
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    val_dataset =  TensorDataset(X_val, y_val)

    #balancing such that same amount of 1&0 
    if BALANCE_METHOD == "undersampling":  #Check balancing method

        class_0_indices = [i for i, label in enumerate(y_train) if label == 0]
        class_1_indices = [i for i, label in enumerate(y_train) if label == 1]

        # Undersample to the size of the smaller class
        min_class_size = min(len(class_0_indices), len(class_1_indices))

        #leads to the fact that the very last items, so project specific stuff from
        #the majority class is cut meaning, meaning we have class_1 data on these 
        #projects, but no class 0 

        random.shuffle(class_0_indices)
        random.shuffle(class_1_indices)

        balanced_indices = class_0_indices[:min_class_size] + class_1_indices[:min_class_size]

        # Create a balanced dataset
        balanced_train_dataset = Subset(train_dataset, balanced_indices)
        train_dataset = balanced_train_dataset

    if (COMBINATION == "VECTOR_ONLY"): 
        input_dim = X_train.shape[1]* X_train.shape[2]
    
    elif (COMBINATION == "VECTOR_AND_IDS"): 
        #TODO
        input_dim = X_train.shape[1]* X_train.shape[2]

    elif (COMBINATION == "ALL_EMBEDDINGS"):
        input_dim = X_train.shape[1]* X_train.shape[2]* X_train.shape[3]

    print ("no error in final creation")
    return train_dataset, test_dataset, val_dataset, input_dim


########## LETS PREPARE THE DATA FOR THE NN ######################
def create_datasets(input_path,directory_path_data, random_seed, sample_subset):

    set_random_seed(random_seed)

    all_data = preprocess_data_construct_pairs(input_path, directory_path_data)

    all_train_node_pairs, all_train_labels, all_test_node_pairs, all_test_labels, all_val_node_pairs, all_val_labels=filter_dataset(all_data, sample_subset)

    train_dataset, test_dataset, val_dataset, input_dim = create_final_dataset(all_train_node_pairs, all_train_labels, all_test_node_pairs,all_test_labels, all_val_node_pairs, all_val_labels)

    

    return train_dataset, test_dataset, val_dataset, input_dim