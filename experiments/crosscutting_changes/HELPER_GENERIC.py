import sys
import os
from typing import Counter

from experiments.crosscutting_changes.HELPER_configuration import TRAIN_RATIO, TRAIN_TEST_VAL_RATIO, VAL_RATIO

current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Get the grandparent directory (parent of the parent)
grandparent_dir = os.path.dirname(parent_dir)

sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)


import random

import os
import torch
import os

def split_train_val_test(graphs): ### WHAT TO CONSIDER TEST; TRAIN ; VAL 
    if (TRAIN_TEST_VAL_RATIO == "ONE_ONLY"): 
        
        # Ensure at least 2 graphs to extract 1 val and 1 test
        if len(graphs) < 3:
            print(f"Skipping  not enough graphs for ONE_ONLY split")
          

        split_index_train = len(graphs) - 2
        split_index_val = len(graphs) - 1
        

    #oder 90,10,10
    elif TRAIN_TEST_VAL_RATIO == "PERCENTAGES" : 
        split_index_train = int(len(graphs) * TRAIN_RATIO) 
        split_index_val = split_index_train + int(len(graphs) * VAL_RATIO)
    
    return split_index_train, split_index_val


def sort_graphs_chrono(graphs):
    sorted_graphs = sorted(graphs, key=lambda g: int(g.diff_id.split("_")[1].split(".")[0]))
    return sorted_graphs


def append_to_pth(file_path, new_pairs, new_labels):
    """
    Incrementally append data to a .pth file.
    If the file does not exist, create it.
    """
    # Prepare tensors
    #new_pairs_tensor = torch.tensor(new_pairs, dtype=torch.float32)
    #new_labels_tensor = torch.tensor(new_labels, dtype=torch.float32)

    #if os.path.exists(file_path):
        # Load existing data
     #   existing_data = torch.load(file_path)
      #  pairs = torch.cat([existing_data["pairs"], new_pairs_tensor])
       # labels = torch.cat([existing_data["labels"], new_labels_tensor])
    #else:
        # Initialize with new data
     #   pairs = new_pairs_tensor
      #  labels = new_labels_tensor

    # Save updated data
    #torch.save({"pairs": pairs, "labels": labels}, file_path)

    
    if os.path.exists(file_path):
        existing_data = torch.load(file_path)
        all_pairs = existing_data["pairs"] + new_pairs  # list concatenation
        all_labels = existing_data["labels"] + new_labels
    else:
        all_pairs = new_pairs
        all_labels = new_labels

    torch.save({
        "pairs": all_pairs,   # List of 5-element tuples
        "labels": all_labels  # Corresponding labels, if separate
    }, file_path)


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_split_counts(all_data):
    all_train_labels = []
    all_val_labels = []
    all_test_labels = []

    for subfolder in all_data:
        all_train_labels.append(all_data[subfolder]["train"]["labels"])
        all_val_labels.append(all_data[subfolder]["val"]["labels"])
        all_test_labels.append(all_data[subfolder]["test"]["labels"])

    print_distribtutions(all_train_labels, all_test_labels, all_val_labels)


def print_distribtutions(all_train_labels,all_test_labels, all_val_labels):
    #THIS IS ONLY FOR PRINTING

    # Flatten inner lists before concatenation
    flat_train_labels = torch.cat([torch.tensor(l) if not isinstance(l, torch.Tensor) else l for l in all_train_labels]).tolist()
    flat_test_labels = torch.cat([torch.tensor(l) if not isinstance(l, torch.Tensor) else l for l in all_test_labels]).tolist()

    print("Training Label Distribution:", Counter(flat_train_labels))
    print("Testing Label Distribution:", Counter(flat_test_labels))

    if all_val_labels:
        flat_val_labels = torch.cat([torch.tensor(l) if not isinstance(l, torch.Tensor) else l for l in all_val_labels]).tolist()
    else:
        flat_val_labels = []
    print("Validation Label Distribution:", Counter(flat_val_labels))

    total = len(flat_train_labels) + len(flat_test_labels) + len(flat_val_labels)
    print(f"Total samples: {total}")
    print(f"Train: {len(flat_train_labels)} ({len(flat_train_labels)/total:.2%})")
    print(f"Val:   {len(flat_val_labels)} ({len(flat_val_labels)/total:.2%})")
    print(f"Test:  {len(flat_test_labels)} ({len(flat_test_labels)/total:.2%}) \n")


    
def make_output_folder(base_path, TRAIN_TEST_SPLIT, BATCH_SIZE,HIDDEN_LAYERS, LOSSFUNCTION, LOSS_FOCAL_ALPHA, LOSS_FOCAL_GAMMA, LOSS_FOCAL_MISCLASS_PENALTIY, POS_WEIGHT_FACTOR, LEARNING_RATE, NUMBER_EPOCHS):
    folder_name = f"nnout_" \
                  f"split-{TRAIN_TEST_SPLIT}_" \
                  f"batch-{BATCH_SIZE}_" \
                  f"layers-{'-'.join(map(str, HIDDEN_LAYERS))}_" \
                  f"loss-{LOSSFUNCTION}_" \
                  f"alpha-{LOSS_FOCAL_ALPHA}_" \
                  f"gamma-{LOSS_FOCAL_GAMMA}_" \
                  f"mispen-{LOSS_FOCAL_MISCLASS_PENALTIY}_" \
                  f"posw-{POS_WEIGHT_FACTOR}_" \
                  f"lr-{LEARNING_RATE}_" \
                  f"epochs-{NUMBER_EPOCHS}"

    output_path_neural_network = os.path.join(base_path, folder_name)
    return output_path_neural_network