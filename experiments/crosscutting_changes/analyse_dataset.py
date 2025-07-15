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




def extract_model_name(filename):
    """
    Given a filename like
    "technology.stem!!models_foodproduction_org.eclipse.stem.foodproduction_model_foodproduction.ecore"
    returns "foodproduction".
    """
    base, _ = os.path.splitext(filename)

    # CHANGED: take only the substring after the last underscore
    segment = base[base.rfind('_') + 1:]    # “…_foodproduction” → “foodproduction”

    # CHANGED: remove any stray underscores
    final_name = segment.replace('_', '')

    return final_name

input_path ="../dataset_raw/diffgraph_new_all/"
name_sub_folders= "default" #"/diffgraphs/"

import statistics

# --- just before your main loop, initialize a place to collect per‐folder stats:
folder_stats = {}

# CHANGED: init global counters
all_commits = []           # one entry per graph
all_total_nodes = []       # one entry per graph
all_total_changes = []     # one entry per graph


# … your existing “for folder_name in os.listdir(input_path):” loop starts here …
for folder_name in os.listdir(input_path):
    input_dir = os.path.join(input_path, folder_name, name_sub_folders)
    if not os.path.isdir(input_dir):
        continue

    graphs = load_components_networkx(data_folder=input_dir, mark_filename=True)
    # prepare lists to collect counts for this folder
    changed_counts = []
    deleted_counts = []
    added_counts   = []
    preserved_counts = []
    total_nodes_counts = []
    folder_name= extract_model_name(folder_name)

    for graph in graphs:
        # — your existing per‐graph node‐classification code —
        all_nodes = []
        preserve_nodes = set()
        deleted_nodes  = set()
        added_nodes    = set()
        changed_nodes  = set()

        for node, data in graph.nodes(data=True):
            try:
                nd = ast.literal_eval(clean_up_string_ast_literal(data['label']))
            except (ValueError, SyntaxError):
                all_nodes = []
                break
            ct = nd.pop('changeType', None)
            if   ct == 'Preserve':    preserve_nodes.add(node)
            elif ct in ('Deleted','Delete','Remove'): deleted_nodes.add(node)
            elif ct == 'Change':      changed_nodes.add(node)
            elif ct == 'Add':         added_nodes.add(node)
            else:
                raise ValueError(f"Unexpected changeType {ct!r}")
            all_nodes.append(node)

        # record counts (skip empty/skipped graphs)
        if all_nodes:
            changed_counts.append(len(changed_nodes))
            deleted_counts.append(len(deleted_nodes))
            added_counts.append(len(added_nodes))
            preserved_counts.append(len(preserve_nodes))
            total_nodes_counts.append(len(all_nodes))
            # CHANGED: per-graph totals for global stats
            total_change = len(changed_nodes) + len(deleted_nodes) + len(added_nodes)
            all_commits.append(1)  # one per graph
            all_total_nodes.append(len(all_nodes))
            all_total_changes.append(total_change)

    # save folder‐level stats
    n = len(changed_counts)
    folder_stats[folder_name] = {
        'Graphs'       : len(graphs),
        'Avg Changed'  : statistics.mean(changed_counts)    if n else 0,
        'Avg Deleted'  : statistics.mean(deleted_counts)    if n else 0,
        'Avg Added'    : statistics.mean(added_counts)      if n else 0,
        'Avg Preserved': statistics.mean(preserved_counts)  if n else 0,
        'Avg Total'    : statistics.mean(total_nodes_counts) if n else 0,
       
    }

# --- after the loop, emit the LaTeX table:
print(r"\begin{tabular}{lrrrrrr}")
print(r"\toprule")
print(r"Dataset & Graphs & Avg Changed & Avg Deleted & Avg Added & Avg Preserved & Avg Total Nodes \\")
print(r"\midrule")
for ds, stats in folder_stats.items():
    print(f"{ds} & "
          f"{stats['Graphs']} & "
          f"{stats['Avg Changed']:.2f} & "
          f"{stats['Avg Deleted']:.2f} & "
          f"{stats['Avg Added']:.2f} & "
          f"{stats['Avg Preserved']:.2f} & "
          f"{stats['Avg Total']:.2f} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")


print("\n\n% Overall Statistics")
print(r"\begin{tabular}{lcccc}")
print(r"\toprule")
print(r"& Projects & Commits & Changes / Commit & Graph Size (nodes) \\")  # CHANGED
print(r"\midrule")
print(f"Value & "
      f"{len(folder_stats)} & "
      f"{np.sum(all_commits):.0f} & "
      f"{np.mean(all_total_changes):.2f} & "
      f"{np.mean(all_total_nodes):.2f} \\\\")  # CHANGED
print(r"\bottomrule")
print(r"\end{tabular}")