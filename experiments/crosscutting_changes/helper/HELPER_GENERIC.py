import json
import re
import sys
import os
from typing import Counter

import psutil



current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Get the grandparent directory (parent of the parent)
grandparent_dir = os.path.dirname(parent_dir)

sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)

from experiments.crosscutting_changes.helper.HELPER_configuration import TRAIN_RATIO, TRAIN_TEST_VAL_RATIO, VAL_RATIO
from networkx.readwrite import json_graph
import random
import networkx as nx
import os
import torch
import os

def _graph_to_dict_general(graph, graph_id):
    data = {
        "id": graph_id,
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": [{"id": str(node), **{key: str(value) for key, value in attr.items()}} for node, attr in graph.nodes(data=True)],
        "links": [{"source": str(edge[0]), "target": str(edge[1]), **{key: str(value) for key, value in edge[2].items()}} for edge in graph.edges(data=True)]
    }
    return data
    
def save_components_networkx(graph: nx.Graph, path_results: any, filename: str):
    """Serialize as nx Json 

    Args:
        graphs (List[nx.Graph]): _description_
        path_results (pathlib.Path): _description_
    """
    os.makedirs(path_results, exist_ok=True)
   
    graph_dicts = _graph_to_dict_general(graph, graph.diff_id)

    json_file_path = os.path.join(path_results,filename)

    with open(json_file_path, 'w') as f:
        json.dump(graph_dicts, f, indent=4)

    
   
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


def load_components_networkx(data_folder, mark_filename = False):
    components = []
    for filename in os.listdir(data_folder):
        if not filename.endswith('.json'):
            continue
        with open(os.path.join(data_folder, filename), 'r') as f:  # open in readonly mode
            json_str = f.read()
            data = json.loads(json_str)
            f.close()
            H = json_graph.node_link_graph(data)
            if (mark_filename): 
                H.diff_id=filename
            components.append(H)
    return components



def print_distribtutions(all_train_labels,all_test_labels, all_val_labels):
    #THIS IS ONLY FOR PRINTING

    # Flatten inner lists before concatenation
    #flat_train_labels = torch.cat([torch.tensor(l) if not isinstance(l, torch.Tensor) else l for l in all_train_labels]).tolist()
    #flat_test_labels = torch.cat([torch.tensor(l) if not isinstance(l, torch.Tensor) else l for l in all_test_labels]).tolist()
    if all_train_labels:
        flat_train_labels = torch.cat([torch.tensor(l) if not isinstance(l, torch.Tensor) else l for l in all_train_labels]).tolist()
    else:
        flat_train_labels = []

    if all_test_labels:
        flat_test_labels = torch.cat([torch.tensor(l) if not isinstance(l, torch.Tensor) else l for l in all_test_labels]).tolist()

    else:
        flat_test_labels = []
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

def save_to_json(path_results, file_name, elements):
    os.makedirs(path_results, exist_ok=True)
    output_path = os.path.join(path_results, f"{file_name}.json")

    with open(output_path, "w") as output_file:
        json.dump(elements, output_file, indent=2)

def clean_up_string_ast_literal(input: str) -> str:
    input = input.replace('\\\'', '"') # used for quotes in strings

    #But your final format is for ast.literal_eval, which expects single quotes for strings (not JSON-style).
    #Worse, it may leave things like ''<p>... partially converted, causing invalid syntax like ''<p>\\r\\n....
    input = re.sub('\'\'([^\']+)\'\'', '"\\1"', input) # double single quotes also used for quotes in strings
    #input = re.sub('\'([\':,]*)\'(?![,\]\}\:])', '\\1', input)




     # word or strings surrounded by quotes and whitespaces (or other ending characters not including default json stuff such as brackets, colons or commata).
     # f you want to remove the quotes around certain strings without removing the character following the closing quote, you should modify the regex to use a lookahead assertion. This way, the regex will check for the condition without consuming the character, which means it won't be replaced or removed during the substitution.
    input = re.sub('\s\'([\w\s\.,-]*)\'(?=[^,\]}:])', '\\1', input)


    input = re.sub('\'(\\w+\(\))\'', '\\1', input) # sometimes method names are put in quotes
    
    
    input = re.sub('\'(\\w+)\'\\\\\\\\nVersion .', '\\1', input) # one-off thing, don't know how this string actuall is created but it's there, so we have to handle it.
    input = re.sub('\'(\\w+)\'\\\\nVersion .', '\\1', input) # one-off thing, don't know how this string actuall is created but it's there, so we have to handle it.
    input = re.sub('\'\\\\\\\\nVersion .', '', input) # one-off thing, don't know how this string actuall is created but it's there, so we have to handle it.
    input = re.sub('\'s ', 's ', input) # one-off thing
    input = re.sub(r"(?<=[a-zA-Z])'s", r"s", input)
    
    input = re.sub('\'stack\' ', 'stack ', input) # one-off thing
    input = re.sub(' \'instructions\' ', ' instructions ', input) # one-off thing
    input = re.sub('\'MIME: \'', 'MIME:', input) # one-off thing
    input = re.sub('\'MIME:', 'MIME:', input) # one-off thing
    input = re.sub('\'selected\' ', 'selected ', input) # one-off thing
    input = re.sub(': \'ecore::EDoubleObject\'', ': ecore::EDoubleObject', input) # one-off thing
    input = re.sub('\'in\' ', 'in ', input) # one-off thing
    input = re.sub('_\'in\'', '_in', input) # one-off thing
    input = re.sub('\s\'in\'', 'in', input) # 'in', 'inout', 'out', pr 'return'
    input = re.sub('\s\'inout\'', 'inout', input) # 'in', 'inout', 'out', pr 'return'
    input = re.sub('\s\'out\'', 'out', input) # 'in', 'inout', 'out', pr 'return'
    input = re.sub('\'out_', 'out_', input) # 'in', 'inout', 'out', pr 'return'

    input = re.sub('\s\'return\'', 'return', input) # 'in', 'inout', 'out', pr 'return'
    input = re.sub('_\'context\'', '_context', input) # one-off _context
    input = re.sub('\'\*\'', '\*', input) # one-off '*'
    input = re.sub(' \'alt\'', ' alt', input) # one-off 'alt'
    input = re.sub('_\'conte', '_conte', input) # one-off _'conte

    input = re.sub('_\'body\'', '_body', input) # one-off _'body'
    input = re.sub('\._\'', '._', input) # one-off ._'

    input = re.sub('\'::\'', '::', input) # 'in', 'inout', 'out', pr 'return'
    input = re.sub('\'create\'(?![,\]\}\:])', 'create', input) # 'in', 'inout', 'out', pr 'return'
    input = re.sub('\'ignore\'(?![,\]\}\:])', 'ignore', input) # 'in', 'inout', 'out', pr 'return'

    input = input.replace('\\r', '\\\\r')
    input = input.replace('\\n', '\\\\n')
    input = input.replace('\\t', '\\\\t') 

  
   # pattern_value = r"(value':\s*')(.*?)(\}+)"

    #def replacer(m):
        # m.group(1) => "value': "
        # m.group(2) => one or two single quotes right after value':
        # m.group(3) => the middle text (no single quotes, thanks to [^']*)
        # m.group(4) => one or two single quotes right before the braces
        # m.group(5) => the closing brace(s), e.g. "}}" or "}" etc.

        # Remove ALL single quotes from the middle text.
       # cleaned_middle = m.group(2).replace("'", "")

        # Reassemble. We unify the outer quotes to a single quote on each side.
        # So even if the original had two quotes (''), it now becomes one (').
      #  return f"{m.group(1)}{cleaned_middle}'{m.group(3)}"

    #input = re.sub(pattern_value, replacer, input)
    return input




def log_resource_usage(epoch, model, log_path, note=""):
    process = psutil.Process(os.getpid())

    cpu_usage = psutil.cpu_percent(interval=None)
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 ** 2)

    # Zeit auf GPU kann auf MPS nicht exakt gemessen werden, daher nur Hinweis
    device_name = next(model.parameters()).device

    log_message = f"[Epoch {epoch}] {note}\n" \
                  f"  CPU usage: {cpu_usage:.2f}%\n" \
                  f"  RAM usage: {memory_mb:.2f} MB\n" \
                  f"  Model device: {device_name}\n"

    if device_name.type == "cpu":
        log_message += " Model is on CPU – GPU not used!\n"
    elif device_name.type == "mps":
        log_message += " Model is using Apple MPS GPU\n"
    elif device_name.type == "cuda":
        log_message += " Model is using CUDA GPU\n"

    print(log_message)
    with open(log_path, "a") as f:
        f.write(log_message + "\n")