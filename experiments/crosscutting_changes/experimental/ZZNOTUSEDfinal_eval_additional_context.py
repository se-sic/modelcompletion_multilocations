# DEBUG STAGE 90/100
import os
import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Get the grandparent directory (parent of the parent)
grandparent_dir = os.path.dirname(parent_dir)
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

# Add the grandparent directory to sys.path
sys.path.insert(0, grandparent_dir)

from experiments.crosscutting_changes.helper.HELPER_final_eval import compute_precision, load_final_results



import pandas as pd
from itertools import combinations

def additional_context_precision(df, top_k=5):
    # Split meta column
    df[['graph_id', 'node_id', 'node_id_pred', 'distance']] = (
        pd.DataFrame(df['meta'].tolist(), index=df.index)
    )
     # --- 2) rank within each source node -----------------------------------
    #df['rank'] = (
     #   df.groupby(['graph_id', 'node_id'])['probability']
      #    .rank(method='first', ascending=False)
    #)

    # Prepare results
    new_ranks = []
    new_precisions = []

    for (gid, node_id), group in df.groupby(['graph_id', 'node_id']):
        # group: all top-k predictions from node_id in graph gid
        # Check which have label == 1
        
        #this was actually the last element
        correct_items = group[group["label"]==1]  # get the items where we know x-> y change together 
        #predict whether z changes with y 
        # based on x, y so  we want to precit  y->z , based on  y->z probabilty and x-> z probalbitly 
        for _, row in correct_items.iterrows():

            pred_node = row["node_id_pred"] # this is y

            # Now find rows in df where node_id == x and node_id_pred == z
            x_to_z_mask = (df['graph_id'] == gid) & (df['node_id'] == node_id)
            # key = z
            x_to_z_probs = df.loc[x_to_z_mask, ["node_id_pred", "probability"]].set_index("node_id_pred")  


            #this is what we want to predict 
            y_to_z_mask = (df['graph_id'] == gid) & (df['node_id'] == pred_node)

            for idx in df[y_to_z_mask].index:
                z = df.at[idx, "node_id_pred"]
                if z in x_to_z_probs.index:
                    #this can be > 1 element, since if 
                    # in graph x -> z -> a, x -> z -> b , with a and b being the new nodes
                    # so mean makes sense 
                    context_prob = x_to_z_probs.loc[z, "probability"].max()

                    df.at[idx, "combined_sum"] = df.at[idx, "probability"] + context_prob
                    df.at[idx, "propagated_context_prob"] = context_prob
          
    #→ Filters out rows where combined_sum is NaN and creates a copy of the result.
    df_combined_sum = df[df['combined_sum'].notna()].copy()
    
    group_sizes, precisions_combined ,precisions_allc = compute_precision(df_combined_sum, top_k=5, score_col='combined_sum')
    group_sizes, precisions , precisions_alla = compute_precision(df_combined_sum, top_k=5, score_col='probability')
    
    plt.figure(figsize=(8, 5))

# Plot two violins
    plt.violinplot([precisions_allc, precisions_alla], showmedians=True)

    # Customize x-axis
    plt.xticks([1, 2], ["additional_context", "anchor_node_only"])

    plt.title("Precision@k with and without additional context")
    plt.ylabel("Precision@k")
    plt.xlabel("Scoring Method")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
        



file_path = "../Final_eval/"
#file_path="../output_dataset_label/eval/nn/"
df = load_final_results(file_path)

additional_context_precision(df)