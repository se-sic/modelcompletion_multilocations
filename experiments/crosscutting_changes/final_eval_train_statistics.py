
import os
import sys


import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Get the grandparent directory (parent of the parent)
grandparent_dir = os.path.dirname(parent_dir)
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

# Add the grandparent directory to sys.path
sys.path.insert(0, grandparent_dir)

from experiments.crosscutting_changes.HELPER_final_eval import load_final_results, prune_leaf_edges


def plot_distance_frequency_by_label(df, plot=True,target_distances=[ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    import pandas as pd
    import matplotlib.pyplot as plt
    df = df[df["approach"] == "NeuralNetwork"]

    # Compute counts of each target distance by label
    # --- Explode and unpack meta ----------------------------------  
    df_exploded = df.explode(['meta', 'label', 'probability']).reset_index(drop=True)
    df_exploded[['graph_id', 'node_id', 'node_id_pred', 'distance']] = pd.DataFrame(df_exploded['meta'].tolist())
    counts = []
    for d in target_distances:
        subset = df_exploded[df_exploded['distance'] == d].groupby('label').size() 
       
        tp = subset.get(1, 0)
        fp = subset.get(0, 0)
        total = tp + fp
        pct = 100 * tp / total if total > 0 else 0  # CHANGED
        counts.append({
            'distance': str(d),
            'false-positive': fp,
            'true-positive': tp,
            'true-positive (%)': round(pct, 1)       # CHANGED
        })

    # CHANGED: Handle float('inf') separately
    subset_inf = df_exploded[df_exploded['distance'] == float('inf')].groupby('label').size()
    if not subset_inf.empty:
        tp = subset_inf.get(1, 0)
        fp = subset_inf.get(0, 0)
        total = tp + fp
        pct = 100 * tp / total if total > 0 else 0  # CHANGED
        counts.append({
            'distance': 'inf',
            'false-positive': fp,
            'true-positive': tp,
            'true-positive (%)': round(pct, 1)       # CHANGED
        })
    df_counts = pd.DataFrame(counts)
    df_counts = df_counts[['distance', 'true-positive', 'false-positive', 'true-positive (%)']]  # CHANGED


    # Plot
    if plot:
        fig, ax = plt.subplots()
        df_counts.set_index('distance')[['true-positive', 'false-positive']].plot(kind='bar', ax=ax)  # CHANGED


        ax.set_ylabel("Count")
        ax.set_xlabel("Distance value")
        ax.set_title("Frequency of specific distances by label")
        ax.legend(title="Label")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()
   # plt.close()


    df_counts = df_counts[['distance', 'true-positive', 'false-positive', 'true-positive (%)']]
    latex_table = df_counts.to_latex(index=False, caption="True/False positives by distance", label="tab:distance_stats")

    print(latex_table)
    return df_counts



if __name__ == "__main__":
    file_path = "../Final_eval/output_ONLYONE4/"



    df = load_final_results(file_path)
    df = prune_leaf_edges(df) 




    plot_distance_frequency_by_label(df)