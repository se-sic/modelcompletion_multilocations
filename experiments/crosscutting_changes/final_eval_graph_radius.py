from collections import defaultdict
import math
import os
import sys

import numpy as np

import numpy as np
import matplotlib.pyplot as plt
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Get the grandparent directory (parent of the parent)
grandparent_dir = os.path.dirname(parent_dir)
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

# Add the grandparent directory to sys.path
sys.path.insert(0, grandparent_dir)

from experiments.crosscutting_changes.final_eval_train_statistics import plot_distance_frequency_by_label

from experiments.crosscutting_changes.HELPER_final_eval import add_node_level_ap, compute_precision, flatten_if_nested, flatten_to_1d, load_final_results, prune_leaf_edges

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr

#DONE NEW wir filtern bestimmte radi in der besprechnung raus 
#max_dist None wenn inf rein 
def filter_radius(df: pd.DataFrame, min_dist: float = 0, max_dist=None, onlyMax=False):
    """
    Filters exploded data by distance and optionally removes 'inf'.
    - Uses meta fields for filtering.
    - Drops ['graph_id', 'node_id', 'node_id_pred', 'distance'] before returning.
    - Does NOT modify the original df.
    """

    # --- Filter only NN approach ----------------------------------
    df = df[df["approach"] == "NeuralNetwork"].copy()

    # --- Explode all relevant columns -----------------------------
    df_exploded = df.explode(['meta', 'label', 'probability'])
    df_exploded["orig_index"] = df_exploded.index  #CHANGED
    meta_cols = ['graph_id', 'node_id', 'node_id_pred', 'distance']
    # --- Expand meta fields ---------------------------------------

    df_exploded = df_exploded.reset_index(drop=True)
    meta_df = pd.DataFrame(df_exploded['meta'].tolist(), columns=meta_cols)
    for col in meta_cols:
        df_exploded[col] = meta_df[col]


    # --- Distance range filter -------------------------------------
    df_exploded = df_exploded[df_exploded['distance'] >= min_dist].copy()
    if max_dist is not None:
        df_exploded = df_exploded[df_exploded['distance'] <= max_dist].copy()

    if onlyMax: 
        df_exploded = df_exploded[df_exploded['distance'] == max_dist].copy()

    # --- Step 4: Keep only original rows where *some* entry passed ---
    group_cols = ['meta', 'label', 'probability']
    first_cols = [c for c in df_exploded.columns if c not in group_cols + ['orig_index']]

    agg_dict = {col: list for col in group_cols}
    agg_dict.update({col: 'first' for col in first_cols})

    df_grouped = df_exploded.groupby("orig_index").agg(agg_dict).reset_index(drop=True)
    return df_grouped
    
   


#DONE approach S: target radius, so if we just increas the set of target labels
def plot_precision_vs_radius_target_v2(df, ks=(0, 1, 2, 3), top_k=5):
    """
    Plots how precision@k changes as the radius increases.
    Uses the `compute_precision` function for precision calculation.

    Parameters:
    - df: DataFrame that must include: ['meta', 'label', 'probability', 'key', 'radius_1_flags', 'radius_2_flags', 'radius_3_flags']
    - ks: Radii to test
    - top_k: Value of k for top-k precision
    """
    records = []
    df = df[df["approach"] == "NeuralNetwork"]
    df_expanded = df.explode(
    ['meta', 'label', 'probability', 'radius_1_flags', 'radius_2_flags', 'radius_3_flags'],
    ignore_index=True
)

# Step 2: Unpack the tuples inside 'meta' into new columns
    df_expanded[['graph_id', 'node_id', 'node_id_pred', 'distance']] = pd.DataFrame(
        df_expanded['meta'].tolist(), index=df_expanded.index
    )
  

    for approach, df_group in df_expanded.groupby("approach"): 

        for r in ks:
            if r == 0:
                flag_col = "label"
            else:
                flag_col = f"radius_{r}_flags"

            #TODO priecison aufräumen berechnet moment per graph avg that is not good
            project_size, avg_precision_per_project, node_prec_df = compute_precision(df_expanded, ground_Truth_colomn_name=flag_col, score_col="probability",  top_k=top_k)
            for _, row in node_prec_df.iterrows():          # one record per (graph_id,node_id)
                records.append({
                    "radius": r,
                    "precision": float(row["precision_at_k"])
                })
            #records.append({"radius": r, "approach": approach, "precision": avg_precision_per_project})

    prec_df = pd.DataFrame(records)

    # ---------- plotting ----------
    plt.figure(figsize=(813))

    ax = sns.violinplot(
        data=prec_df,
        x="radius",
        y="precision",
        inner=None,
        cut=0,
        scale="width"
    )

    sns.stripplot(
        data=prec_df,
        x="radius",
        y="precision",
        color="black",
        jitter=True,
        size=3,
        alpha=0.6
    )

    # ── Mean per radius (red dots) ─────────────────────────────────────
    group_means = prec_df.groupby("radius")["precision"].mean()
    for i, r in enumerate(sorted(group_means.index)):
        ax.scatter(i, group_means[r], color="red", s=50, zorder=10, marker="o")

    plt.ylim(0, 1.05)
    plt.xlabel("Radius r")
    plt.ylabel(f"Precision@{top_k}")
    plt.title(f"Precision@{top_k} per radius")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
#HELPER function of radi_dependence
def plot_radius_means_lineplot(keys, df_long, metric_col, radii):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    df_long = df_long.copy()

    # Convert radii column to strings for categorical x-axis
    df_long["RadiusStr"] = df_long["Radius"].apply(lambda r: '∞' if r == 'inf' or r == float("inf") else str(int(r)))
    radii_str = ['∞' if r == 'inf' or r == float("inf") else str(int(r)) for r in radii] 

    plt.rcParams.update({  # CHANGED
    'font.size': 15,
    'axes.labelsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
    'axes.titlesize': 15
})  # CHANGED
    
    plt.figure(figsize=(13, 5))
    ax = plt.gca()
    palette = sns.color_palette("tab10", n_colors=len(keys))

    for idx, k in enumerate(keys):
        data_k = df_long[df_long["k"] == k].copy()

        if '1' in radii_str: 
            data_k = pd.concat([data_k, pd.DataFrame({
                "RadiusStr": ['1'], "k": [k], metric_col: [0]
            })])

        means = data_k.groupby("RadiusStr")[metric_col].mean()

        x = []
        y = []
        for r_str in radii_str:
            if r_str in means.index:
                x.append(r_str)
                y.append(means[r_str])

        ax.plot(x, y, marker="o", markersize=8, alpha=0.7, label=f"k={idx+1}", color=palette[idx], linewidth=1.2)
    df_baseline = plot_distance_frequency_by_label(df, plot=False)                      
    df_baseline["distance_num"] = df_baseline["distance"].apply(                        
        lambda x: float("inf") if x == "inf" else int(x))                                    

    cum_probs = []                                                                      
    for r_str in radii_str:                                                            
        r_num = float("inf") if r_str == '∞' else int(r_str)                           
        mask  = df_baseline["distance_num"] <= r_num                                   
        tp    = df_baseline.loc[mask, "true-positive"].sum()                           
        fp    = df_baseline.loc[mask, "false-positive"].sum()                          
        cum_probs.append(tp / (tp + fp) if (tp + fp) else 0)                            

    ax.plot(radii_str, cum_probs, marker='s',markersize=8, linestyle='--', color='black',             
            label='p(TP)', linewidth=1.2)    


    ax.set_xlabel("Radius")
    ax.set_ylabel(f"Mean {metric_col}")
    #ax.set_title(f"Mean {metric_col} vs Radius")
    ax.set_xticks(radii_str) 
    ax.set_xticklabels(radii_str) 
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Precision@k", bbox_to_anchor=(1.02, 1), loc="upper left") 
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.expanduser("~/Downloads/radius_plot.pdf"), format="pdf", bbox_inches="tight")
    plt.show()

    #NEW
    print("xxxxxxxxxxxxxxxxxxx")
    radi_dependence_sp_test(df_long, metric_col,
                        cum_probs_by_radius_str=list(zip(radii_str, cum_probs)))  


   




#HELPER function of radi_dependence
def plot_radi_dependence(keys,df_long, metric_col, radii): 
        # 2) violin plots – one hollow outline per k, overlaid 
        # 
    plt.rcParams.update({  # CHANGED
    'font.size': 15,
    'axes.labelsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
    'axes.titlesize': 15
})  # CHANGED
    
    plt.figure(figsize=(13, 5))
    ax = plt.gca()                                                      
    palette = sns.color_palette("tab10", n_colors=len(keys))            

    for idx, k in enumerate(keys):                                       
        data_k = df_long[df_long["k"] == k].copy()
        if 1 in radii:
            data_k = pd.concat([data_k, pd.DataFrame({"Radius": [1], "k": [k], metric_col: [0]})])
        metric_int = idx+1   




        sns.violinplot(                                                  
            data=data_k,
            x="Radius",
            y=metric_col,
            order=radii,                    
            cut=0.0,                    # small tail → rounded tips     
            bw_adjust=.5,               # smoother outline            
            inner=None,
            linewidth=1.2,              # thinner line                  
            fill=False,                                                   
            color=palette[idx],
            scale="width",
            dodge=False,        
        )

        # overlay mean dot for this k                                    
        means_k = data_k.groupby("Radius")[metric_col].mean()          
        for r, m in means_k.items():                                   
            x_pos = list(radii).index(r)                                
            ax.scatter(x_pos, m,                                       
                       color=palette[idx], s=50, zorder=10,
                       label=f"k={metric_int}" if x_pos == 0 else "")

    # legend outside on the right                                      
    handles, labels = ax.get_legend_handles_labels()                     
    by_label = dict(zip(labels, handles))                                
    ax.legend(by_label.values(), by_label.keys(),                        
              title="Precision@k", bbox_to_anchor=(1.02, 1),
              loc="upper left", borderaxespad=0,
              frameon=True, fancybox=False)
    ax.set_xticklabels(['∞' if x=='inf' else int(x) for x in radii])  
    # cosmetics  
    ax.set_ylim(bottom=0)                                                           
    plt.title(f"Distribution of {metric_col} per Radius")    
    plt.ylabel(metric_col)                                               
    plt.grid(axis="y", linestyle="--", alpha=0.5)                        
    plt.tight_layout(rect=[0, 0, 0.85, 1])                               
    plt.show()                                                           

def radi_dependence_sp_test(df_long, metrics_col, cum_probs_by_radius_str=None): 
    
    import numpy as np                                          
    from scipy.stats import spearmanr                           

    df_test = df_long.copy()                                    
    df_test["Radius"] = df_test["Radius"].apply(                
        lambda r: np.inf if r == "inf" else r                   
    )                                                           

    # per‐k trend test                                         
    for k in sorted(df_test["k"].unique()):                     
        sub = df_test[df_test["k"] == k].dropna(                
            subset=["Radius", metrics_col]                     
        )                                                       
        finite = sub[np.isfinite(sub["Radius"])]                
        rho, pval = spearmanr(finite["Radius"],                 
                              finite[metrics_col])              
        print(f"Precision@{k}: ρ = {rho:.3f}, p = {pval:.3g}")  

    # overall mean‐trend test                                   
    mean_df = (df_test
               .groupby("Radius")[metrics_col]
               .mean()
               .reset_index())                                  
    finite = mean_df[np.isfinite(mean_df["Radius"])]            
    rho_all, p_all = spearmanr(finite["Radius"],                 
                               finite[metrics_col])               
    print(f"\nOverall mean trend: ρ = {rho_all:.3f}, p = {p_all:.3g}")  


    #LINK dependence 
    if cum_probs_by_radius_str is not None:  
        print("\n== Lift@k vs Radius ==")
        # build lookup: RadiusStr → p(r)
        prevalence = dict(cum_probs_by_radius_str) 

        df_test["RadiusStr"] = df_test["Radius"].apply(
            lambda r: '∞' if r == np.inf else str(int(r))
        )
        df_test["Lift"] = df_test.apply(
            lambda row: row[metrics_col] / prevalence.get(row["RadiusStr"], np.nan)
            if prevalence.get(row["RadiusStr"], 0) > 0 else np.nan,
            axis=1
        )

        for k in sorted(df_test["k"].unique()):
            sub = df_test[df_test["k"] == k].dropna(subset=["Radius", "Lift"])
            finite = sub[np.isfinite(sub["Radius"])]
            rho, pval = spearmanr(finite["Radius"], finite["Lift"])
            print(f"Lift@{k}: ρ = {rho:.3f}, p = {pval:.3g}")

        mean_lift_df = (df_test
                        .groupby("Radius")["Lift"]
                        .mean()
                        .reset_index())
        finite = mean_lift_df[np.isfinite(mean_lift_df["Radius"])]
        rho_all, p_all = spearmanr(finite["Radius"], finite["Lift"])
        print(f"Overall mean trend in Lift: ρ = {rho_all:.3f}, p = {p_all:.3g}")

       #TODO Δ Precision@k = Precision@k – p(r) (a
        print("\n== ΔPrecision@k vs Radius ==")                     # CHANGED

        df_test["ΔPrecision"] = df_test.apply(                      # CHANGED
            lambda row: row[metrics_col] - prevalence.get(row["RadiusStr"], np.nan)
            if row["RadiusStr"] in prevalence else np.nan,
            axis=1
        )

        for k in sorted(df_test["k"].unique()):                     # CHANGED
            sub = df_test[df_test["k"] == k].dropna(subset=["Radius", "ΔPrecision"])
            finite = sub[np.isfinite(sub["Radius"])]
            rho, pval = spearmanr(finite["Radius"], finite["ΔPrecision"])
            print(f"ΔPrecision@{k}: ρ = {rho:.3f}, p = {pval:.3g}")   # CHANGED

        mean_delta_df = (df_test                                    # CHANGED
                        .groupby("Radius")["ΔPrecision"]
                        .mean()
                        .reset_index())
        finite = mean_delta_df[np.isfinite(mean_delta_df["Radius"])] # CHANGED
        rho_all, p_all = spearmanr(finite["Radius"], finite["ΔPrecision"])
        print(f"Overall mean trend in ΔPrecision: ρ = {rho_all:.3f}, "
            f"p = {p_all:.3g}")                                   # CHANGED

#dependence radi limited to 1,2,...
def radi_dependence(
    df,
    radii         = None,           # iterable of radii to evaluate
    metric_family = "ap", 
    ks            = range(1, 11),   # k values to track (1…10 by default)
    top_k         = 5,              # top-k for rank filtering inside add_node_level_ap
    min_dist      = 0,
    onlyMax =False,
     plot          = True,
    **filter_kw,                   # extra args forwarded to filter_radius
):

     # add an ∞ bin ---------------------------------------------------------  
    radii_with_inf = list(radii) + ["inf"]                                   

    # ---- decide which metric keys we will track -------------------
    if metric_family == "ap":
        keys = ["ap"] + [f"ap@{k}" for k in ks]
    elif metric_family == "prec":
        keys = [f"prec@{k}" for k in ks]
        metrics_col = "Precision@k"
    elif metric_family == "prec_all":
        metrics_col = "Precision@k"
        keys = [f"prec@{k}_all" for k in ks]
    else:
        raise ValueError("metric_family must be 'ap', 'prec', or 'prec_all'")

     # storage: radius -> metric -> list
    results = {r: {k: [] for k in keys} for r in radii_with_inf}             

    # ---- sweep radii ---------------------------------------------
    for r in radii_with_inf:
        # 1) distance-filtered dataframe (all original columns preserved)
        max_d = None if r == "inf" else r                   
        df_r = filter_radius(
            df,
            min_dist=min_dist,
            max_dist=max_d,
            onlyMax=onlyMax
        )

        # 2) compute node-level metrics
        metrics = add_node_level_ap(df_r, ks=ks)

        # 3) collect chosen metrics
        for k in keys:
            vals = []
            for v in metrics[k]:
                vals.extend(v)
            
            results[r][k] = vals   


   
    records = []                                                     
    for r in radii_with_inf:                                                   
        for k in keys:                                                
            for v in results[r][k]:                                   
                records.append({"Radius": r, "k": k,  metrics_col: v})  

    df_long = pd.DataFrame.from_records(records) 
    radi_dependence_sp_test(df_long, metrics_col)
  
 
    if plot: 
        plot_radius_means_lineplot(keys,df_long,  metrics_col , radii_with_inf)
        #plot_radi_dependence(keys,df_long,  metrics_col , radii_with_inf)
    return results


#in which rank are the recommended items most
"""

def plot_mean_distance_by_rank(df,  ks= range(1, 11)):
    # df has columns: 'approach', 'distance@k' (list of distances for top-k predictions)

    plt.figure(figsize=(10, 6))
    df = df[df["approach"] == "NeuralNetwork"]
    
    for _, row in df.iterrows():  
        probs=flatten_to_1d(row["probabilities"]) 
        labs = flatten_to_1d(row["label"])    
        metas= flatten_if_nested(row["meta"])       


        #TODO here entfernen der existierended
        buckets = defaultdict(list)
        for idx, m in enumerate(metas):
            node_id = m
            buckets[node_id].append(idx)


        for indx in buckets.values():
        # sort these indices by score desc
            idxs.sort(key=lambda i: probs[i], reverse=True)
            labels_sorted = [labs[i] for i in sorted_indices]

            # -- Precision@5  --------------------------------------------------  #CHANGED
        total_pos = sum(labels_sorted)
        if total_pos == 0:                             # kein Relevantes
            continue 

    # --- Compute mean distance per rank ---
    dist_matrix = np.vstack(df[f"distance@{max_k}"])   (use filtered df)
    mean_dists = dist_matrix.mean(axis=0)

    plt.plot(range(1, max_k + 1), mean_dists, label="NeuralNetwork", marker="o")   (label explicitly)

    plt.xlabel("Prediction Rank (k)")
    plt.ylabel("Mean Distance from Origin Node")
    plt.title("Mean Distance of Top-k Predictions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
"""


if __name__ == "__main__":
    file_path = "../Final_eval/output_ONLYONE4/"


    #file_path="../output_dataset_label/eval/nn/"
    df = load_final_results(file_path)

    df = df[df["approach"] == "NeuralNetwork"].copy()
    df = prune_leaf_edges(df) 

    #TODO statistiken aus top x wo kpmmen die her 

    #minimalerradisu: min_dist
    radi_dependence(df, min_dist=1, radii=range(2, 10),onlyMax=False, metric_family="prec_all", filter_inf=True)
    #plot_mean_distance_by_rank(df)
    #Svens stuff 
    #plot_precision_vs_radius_target_v2(df)


    #filter_radius(df, filter_inf=False)
