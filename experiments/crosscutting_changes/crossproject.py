
# idea first 

import json
import os
import random
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Get the grandparent directory (parent of the parent)
grandparent_dir = os.path.dirname(parent_dir)

sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)

from experiments.crosscutting_changes.helper.HELPER_GENERIC import load_components_networkx
from experiments.crosscutting_changes.helper.HELPER_neural_networks_dataloader import graphsToPairs, load_train_test_val_data





# Problem this loading and reloarding usually takes a huge amout of time 
# we so need a solution for that 
# if foldername in train other wise in test 

def construct_pairs_10fold(input_path, output_dir, testing_folder): 

    all_data={}
    name_sub_folders= "/default/"
    
    for folder_name in os.listdir(input_path):

        file_path_train = os.path.join(output_dir, folder_name, "dataset_pairs_train.pth")
        file_path_test = os.path.join(output_dir, folder_name, "dataset_pairs_test.pth")
        file_path_val =  os.path.join(output_dir, folder_name, "dataset_pairs_val.pth")

        if all(os.path.exists(p) for p in [file_path_train, file_path_val, file_path_test]):
            continue

        if not os.path.isdir(input_path + '/' + folder_name):
            continue

        input_dir = input_path + '/' + folder_name + name_sub_folders
        
        graphs = load_components_networkx(data_folder=input_dir, mark_filename=True)


        if (folder_name in testing_folder): 
            
            # Ensure at least 2 graphs to extract 1 val and 1 test
            if len(graphs) < 3:
                print(f"Skipping {folder_name} — not enough graphs for ONE_ONLY split")
                continue

            split_index_train = 0
            split_index_val = 0

        else: # folder_name in testing_folder
             # all in train, no in val, no in test
            split_index_train = len(graphs)
            split_index_val = len(graphs)

        graphsToPairs(graphs,folder_name, all_data, split_index_train, split_index_val, file_path_train, file_path_val, file_path_test) 
          
    return all_data
            

def construct_distribution_10fold(input_path, output_path, seed=42):
    # set random seed for reproducibility
    random.seed(42)

    # get all files from input_path
    all_files = sorted(os.listdir(input_path))
    all_files.remove('.DS_Store')
   
    all_10fold_combinations = {}

    for i in range(10):
        # pick test indices for this fold
        if len(all_files) == 0: 
            break
       
        test_file = random.sample(all_files, 1) 

        testing_folder = test_file
        train_folders = [f for f in all_files if f not in testing_folder]
        # always another file choosen 
        all_files = train_folders

        all_10fold_combinations[i] = {
            "train": train_folders,
            "test": testing_folder, 
            "filepath": f"{output_path}/fold_{i}__{test_file[0]}/"
        }

    save_path = os.path.join(output_path, "10fold_distribution_mapping.json")
    with open(save_path, "w") as f:
        json.dump(all_10fold_combinations, f, indent=2)

    return all_10fold_combinations





#Todo than new name for new distribtuion embedding_data_new_branch_indivudalemb/" "../output_dataset_label/embedding_data_refactored2"

#  all_data = preprocess_data_construct_pairs(input_path, directory_path_data)
# das kann später dann einfach normal gecalled werden wobei directory_path_data dann die dateien sein sollten 



#TODO focus on one speicfic file depending on parameter 


def run_10fold_assigment(foldnumber, input_path, directory_path_data): 
    all_10fold_combinations = construct_distribution_10fold(input_path, directory_path_data, seed=42)
    testing_folder = all_10fold_combinations[foldnumber]['test']
   
    output_dir = all_10fold_combinations[foldnumber]['filepath']
    os.makedirs(output_dir, exist_ok=True) 
    construct_pairs_10fold(input_path, output_dir, testing_folder)

def testloading(output_dir): 

    all_data={}

    # Skip processing if the files already exist
    if os.path.exists(output_dir):
        
        print(f"Files already exist for {output_dir}. Skipping...")
        all_data= load_train_test_val_data(output_dir)
    
    return all_data

def main():

    if len(sys.argv) == 2: 
        try: 
            foldnumber = int(sys.argv[1])
        except ValueError:
            print("Error: foldnumber must be an integer 0–9.")

        if not (0 <= foldnumber <= 9):
            print("Error: foldnumber must be between 0 and 9.")
            sys.exit(1)
            
    else: 
        foldnumber = 0
        print(f"WARNING: no foldnumber given defaulting to {foldnumber}")

   

    print(f"Running with fold number {foldnumber}")

    directory_path_data =  "/scratch/TODO/CrossCutting/neuralnetworkpairs/10cross/"
    input_path = "/scratch/TODO/CrossCutting/dataset_node_embeddings_text-embedding-3-small-with-ids/diffgraphs/"
        
    # for local 
   # directory_path_data =   "../output_dataset_label/embedding_data_crossfold"
# input_path = "../output_dataset_label/dataset_node_embeddings_text-embedding-3-small-with-ids_small/diffgraphs/"

    os.makedirs(directory_path_data, exist_ok=True) 
    run_10fold_assigment(foldnumber, input_path, directory_path_data)

if __name__ == "__main__":
    test_dir = '../output_dataset_label/embedding_data_crossfold/fold_0__eclipse.e4!!bundles_org.eclipse.e4.ui.model.workbench_model_UIElements.ecore/'
    test_dir = '../resultsmajorrevision/datasets_major_revision_sample_10fold/'
    testloading(test_dir)
    main()