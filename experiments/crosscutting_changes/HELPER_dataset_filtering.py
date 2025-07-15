import random
import torch


from collections import defaultdict




def filter_contradictory_pairs_fast(pairs: torch.Tensor, labels: torch.Tensor):
    
    # 1) Group by exact embedding bytes to find duplicates.
    #    We build a dict: embedding_bytes -> list of indices that share that exact embedding.
    pair_dict = defaultdict(list)

    for i in range(len(pairs)):
        emb_bytes = pairs[i][2]
        emb_tensor= torch.tensor(emb_bytes)
        pair_dict[emb_tensor].append(i)

    # 2) Find any embedding that has *both* label=0 and label=1 => contradictory
    contradictory_indices = set()
    for emb_bytes, idx_list in pair_dict.items():
        these_labels = set(int(labels[i]) for i in idx_list)
        if len(these_labels) > 1:
            # Contradiction => exclude all these indices
            contradictory_indices.update(idx_list)

    # 3) Build the final filtered list (exclude contradictory ones)
    keep_indices = [i for i in range(len(pairs)) if i not in contradictory_indices]

    filtered_pairs = [pairs[i] for i in keep_indices]
    filtered_labels = [labels[i] for i in keep_indices]

    return filtered_pairs, filtered_labels


def remove_duplicates(all_train_node_pairs, all_train_labels):
    """
    Given lists of Tensors (pairs, labels), filter out any contradictory duplicates
    from each element in the list.
    """
    new_all_train_node_pairs = []
    new_all_train_labels = []

    for pairs_tensor, labels_tensor in zip(all_train_node_pairs, all_train_labels):
        filtered_pairs, filtered_labels = filter_contradictory_pairs_fast(pairs_tensor, labels_tensor)
        new_all_train_node_pairs.append(filtered_pairs)
        new_all_train_labels.append(filtered_labels)

    return new_all_train_node_pairs, new_all_train_labels


# --- Hilfsfunktion zum Balancieren eines Subfolder-Datensatzes ---
def change_trainset_size(pairs, labels, method , target_size ):
    """
    Passt die Anzahl der (pairs, labels) an, so dass genau 'target_size'
    Elemente zurückgegeben werden.

    - Wenn len(pairs) > target_size: zufälliges Undersampling auf target_size
    - Wenn len(pairs) < target_size: zufälliges Oversampling (mit Zurücklegen)
    """

    # for both ADJUST_TRAIN_SIZE= "MAKEFOLDERSEQUAL" and "CUT_ABOVE" we need to do this 
    current_size = len(pairs)
    if current_size > target_size:
        # UNDERSAMPLING
        indices = random.sample(range(current_size), target_size)
        pairs = [pairs[i] for i in indices]
        labels = [labels[i] for i in indices]
    
    # only if folders should be of equal size
    if (current_size < target_size and method=="MAKEFOLDERSEQUAL" and current_size != 0 ):
        # OVERSAMPLING
        deficit = target_size - current_size
        # Indizes mit Zurücklegen wählen
        oversample_indices = [random.randrange(current_size) for _ in range(deficit)]
        extra_pairs = [pairs[i] for i in oversample_indices]
        extra_labels = [labels[i] for i in oversample_indices]
        # Konkateniere Original + Oversample
        pairs = pairs + extra_pairs
        labels = labels+ extra_labels

    #  SHUFFLE FOR TRAINING HERE 
    # Wenn == target_size, nichts tun

    #shuffle_indices = torch.randperm(pairs.size(0))
    #pairs = pairs[shuffle_indices]
    #labels = labels[shuffle_indices]

    return pairs, labels



import torch
import random

def ensure_train_val_test_ratio(all_train_node_pairs, all_train_labels,
                                  all_val_labels,
                                  all_test_labels):
    """
    Ensures the 80/10/10 split by oversampling train data if needed.
    """

    # Count total number of items
    train_len = sum(len(x) for x in all_train_labels)
    val_len = sum(len(x) for x in all_val_labels)
    test_len = sum(len(x) for x in all_test_labels)
    total = train_len + val_len + test_len

    expected_train = int(0.8 * total)
    if train_len >= expected_train:
        return all_train_node_pairs, all_train_labels

    # Oversample
    deficit = expected_train - train_len

     # Flatten list of pair tuples
    flat_pairs = [pair for batch in all_train_node_pairs for pair in batch]
    flat_labels = [label for batch in all_train_labels for label in batch]

    # Sample with replacement
    indices = [random.randint(0, len(flat_pairs) - 1) for _ in range(deficit)]
    extra_pairs = [flat_pairs[i] for i in indices]
    extra_labels = [flat_labels[i] for i in indices]

    all_train_node_pairs.append(extra_pairs)
    all_train_labels.append(extra_labels)


   

    return all_train_node_pairs, all_train_labels
