import numpy as np
from collections import defaultdict
import pandas as pd


import json
import os




overall_order= ["Random", "Historical", "Semantics", "NeuralNetwork"]
def clean_key(key):
    """Extracts the second-to-last segment if the key is a path, otherwise returns it as is."""
    segments = key.strip().split("/")  # Split by "/"

    if len(segments) > 1:
        return segments[-2]  # Take second-to-last segment if path exists
    else:
        return key  # Return as is if already cleaned


def load_final_results(directory):
    rows = []

    for file in os.listdir(directory):
        if not file.endswith(".json"):
            continue
        approach = os.path.splitext(file)[0]

        with open(os.path.join(directory, file)) as f:
            for key, vals in json.load(f).items():
                if vals.get("precision_avg") == -1:
                    continue

                cleaned = clean_key(key)


                if approach=="Random" or approach=="Historial" or approach=="Semantics":
                    rows.append({
                            "key": cleaned,
                            "approach": approach,
                            "precision_avg": vals.get("precision_avg"),
                            "precision_all": vals.get("precision_all"),
                            "total_true_positives": vals.get("total_true_positives"),
                            "total_predictions": vals.get("total_predictions"),
                            "probability": vals.get("probabilities"),
                            "label": vals.get("labels"),
                             "meta": vals.get("meta"),  
                        })
                    
                else:
                    probs = vals.get("probabilities", [])
                    labs  = vals.get("labels", [])
                    metas = vals.get("meta", [])

                    # --- check equal lengths ------------------------------------
                    if not (len(probs) == len(labs) == len(metas)):
                        raise ValueError(f"Length mismatch for {key} in {file}")
                    rows.append({
                        "key": cleaned,
                        "approach": approach,
                        #TODO this was a bug!!!!!! qucik fix 
                        #old and crrect"precision_avg": vals.get("precision_avg"),
                        "precision_avg": vals.get("precision_avg"),
                        "precision_all": vals.get("precision_all"),
                        "total_true_positives": vals.get("total_true_positives"),
                        "total_predictions": vals.get("total_predictions"),
                        "radius_1_flags" : vals.get("total_predictions_radius_1"),
                        "radius_2_flags" : vals.get("total_predictions_radius_2"),
                        "radius_3_flags" : vals.get("total_predictions_radius_3"),
                        "probability": vals.get("probabilities"),
                        "label": vals.get("labels"),
                        "meta": vals.get("meta"),           # ensure tuple
                    })

        # Group by key and approach, count unique precision_avg values

    return pd.DataFrame(rows)

def _prune_row(row):
        metas  = row["meta"]
        probs  = row["probability"]
        labels = row["label"]
    

        probs = flatten_to_1d(probs)
        labels = flatten_to_1d(labels)
        metas = flatten_if_nested(metas)

        metas_f, probs_f, labels_f = prune_leaf_edges_once(
        metas=metas, probs=probs, labels=labels
        )
        return metas_f, probs_f, labels_f

def prune_leaf_edges(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every row in *df* run `filter_leaf_targets` and
    replace the columns 'meta', 'probability', 'label'
    with the pruned versions.  Order is preserved.
    """
 
    # Apply row-wise and split the tuple into three new columns
    pruned = df.apply(_prune_row, axis=1, result_type="expand")
    pruned.columns = ["meta", "probability", "label"]

    # Overwrite the original columns
    for col in ("meta", "probability", "label"):
        df[col] = pruned[col]

    return df

def prune_leaf_edges_once(metas, probs=None, labels=None):
    """
    Remove every edge (src → dst) whose *dst* is itself a source
    (all tuples belong to one graph).
    """
    src_nodes = {src for graph, src, target, dis in metas}                 #changed

    keep_mask = [target not in src_nodes for graph, src, target, dis in metas]  #changed

    metas_f  = [m for m, k in zip(metas,  keep_mask) if k]
    probs_f  = [p for p, k in zip(probs,  keep_mask) if k]
    labels_f = [l for l, k in zip(labels, keep_mask) if k]
    return metas_f, probs_f, labels_f

def compute_precision(df, score_col='probability', ground_Truth_colomn_name="label", top_k=5):
    """
    Compute precision@k per group ('key'), using per-node adjusted_k,
    where adjusted_k = min(top_k, number of label==1 in full (graph_id, node_id) group)

    Assumes df contains: 'graph_id', 'node_id', 'probability', 'label', 'key'

    Returns:
        group_sizes (List[int]): Number of items considered per group
        precisions (List[float]): Per-group precision@k
    """
    # --- 1) Rank within each (graph_id, node_id) group ---
    df['rank'] = (
        df.groupby(['graph_id', 'node_id'])[score_col]
          .rank(method='first', ascending=False)
    )

    # --- 2) Count positives per (graph_id, node_id) for precision max value
    pos_counts = (
        df[df[ground_Truth_colomn_name] == 1]
        .groupby(['graph_id', 'node_id'])
        .size()
        .rename('num_positives')
    )

    # --- 3) Merge count info back into df ---
    df = df.merge(pos_counts, on=['graph_id', 'node_id'], how='left')
    df['num_positives'] = df['num_positives'].fillna(0).astype(int)

    # --- 4) Compute adjusted_k per row ---
    df['adjusted_k'] = df['num_positives'].apply(lambda n: min(top_k, n) if n > 0 else 0)

    # --- 5) Filter to top adjusted_k entries per node ---
    # top_k_df now contains, for each (graph_id, node_id), the top-adjusted_k predictions
    # (lowest rank values), ready for further evaluation like precision@k.
    def keep_adjusted_top_k(subdf):
        k = int(subdf['adjusted_k'].iloc[0])
        return subdf.nsmallest(k, 'rank') if k > 0 else subdf.iloc[0:0]

    top_k_df = (
        df.groupby(['graph_id', 'node_id'], group_keys=False)
          .apply(lambda group: keep_adjusted_top_k(group)
             .assign(key=group['key'].iloc[0]))
          .sort_values(['graph_id', 'node_id', 'rank'])
          .reset_index(drop=True)
    )

    # --- 6) Compute overall precision ---
    precision_k_all = (top_k_df[ground_Truth_colomn_name] == 1).mean()
    print(f"Overall Precision = {precision_k_all:.3f}")

    node_level_precisions = (
    top_k_df
    .groupby(['graph_id', 'node_id'])
    .agg(
        precision_at_k=(ground_Truth_colomn_name, 'mean'),
        key=('key', 'first')
    )
    .reset_index()
    )

    # --- 7) Per-key precision ---
    project_size = []
    avg_precision_per_project = []

    for key, group_df in node_level_precisions.groupby("graph_id"):
        precision_k = group_df["precision_at_k"].mean()
        group_size = len(group_df)

        avg_precision_per_project.append(precision_k)
        project_size.append(group_size)
        unique_keys = group_df["key"].unique()

        print(f"{key}: Precision@k = {precision_k:.3f}, num elements = {group_size}, keys = {unique_keys}")

    avg_precision = sum(avg_precision_per_project) / len(avg_precision_per_project)
    print(f"\nAverage Precision@k across all files = {avg_precision:.3f}")

    return project_size, avg_precision_per_project, node_level_precisions


def filter_keys_with_all_approaches(df, required_approaches=None):
    if required_approaches is None:
        required_approaches = set(overall_order) - {"Random"}  # drop if only Random present

    # Filter out rows where precision_all has only 1 element
    df = df[df["precision_all"].apply(lambda x: len(x) > 1)]

    grouped = df.groupby("key")["approach"].unique()
    valid_keys = grouped[grouped.apply(lambda x: required_approaches.issubset(set(x)))].index

    return df[df["key"].isin(valid_keys)].copy()


# -------------------------------------------------------------------
# helper – average precision at (optional) cut-off k
# -------------------------------------------------------------------
def average_precision_at_k(labels, k=None):
    labels = np.asarray(labels).flatten()

    # Find where last relevant item is
    if np.any(labels == 1):
        last_pos = np.max(np.where(labels == 1)) + 1
    else:
        return 0.0  # no relevant items

    # Truncate to min(k, last relevant + 1)
    if k is not None:
        labels = labels[:min(k, last_pos)]
    else:
        labels = labels[:last_pos]

    num_relevant = labels.sum()
    prec_hits = np.cumsum(labels) / (np.arange(1, len(labels) + 1))

    return prec_hits[labels == 1].mean() if np.any(labels == 1) else 0.0


 
def flatten_if_nested(x):
    if isinstance(x, list):
        if isinstance(x[0], list) and isinstance(x[0][0], list):
            return [inner for outer in x for inner in outer]  # 3-level to 2-level
        elif isinstance(x[0], list):
            return x  # already 2-level → no action
    return x  # not a list or 1D → no action

def flatten_to_1d(nested):
    """Flatten any nested list to a flat 1D list."""
    result = []
    stack = [nested]

    while stack:
        current = stack.pop()
        if isinstance(current, list):
            stack.extend(reversed(current))  # preserve order
        else:
            result.append(current)

    return result

import random                       # add once at top

def sort_with_random_ties(idxs, probs, rng=random):
    """
    Return `idxs` sorted by descending prob; ties are randomly permuted.
    """
    # 1️ randomise order first …
    idxs = idxs[:]                  # copy to avoid in-place mutation
    rng.shuffle(idxs)               # <- tie-breaker
    # 2️ … then stable-sort by score
    idxs.sort(key=lambda i: probs[i], reverse=True)
    return idxs

def map_per_node(probs, labs, metas, ks=(3, 5, 10)):
    """
    Compute full MAP and MAP@k, grouping by node_id = meta[i][1].

    Returns
    -------
    dict  – keys: 'ap', 'ap@3', 'ap@5', ...


    """

    #if not isinstance(metas, (list, tuple)):
     #   return {"ap": 0.0, **{f"ap@{k}": 0.0 for k in ks}}

    # ---- split indices by node_id ---------------------------------

    probs = flatten_to_1d(probs)
    labs = flatten_to_1d(labs)
    metas = flatten_if_nested(metas)

    #TODO here entfernen der existierended
    buckets = defaultdict(list)
    for idx, m in enumerate(metas):
        node_id = m[1]
    
       
        #all indices belonging to this           # 2nd item of meta tuple
        buckets[node_id].append(idx)

    # ---- AP per bucket --------------------------------------------
    aps_full = []                                       
    prec_lists = {k: [] for k in ks}     
    aps_k = {k: [] for k in ks}

    for indx in buckets.values():
        # sort these indices by score desc
        sorted_indices = sort_with_random_ties(indx, probs)   
        labels_sorted = [labs[i] for i in sorted_indices]

          # -- Precision@5  --------------------------------------------------  #CHANGED
        total_pos = sum(labels_sorted)
        if total_pos == 0:                             # kein Relevantes
            continue  
        
                           
       
        # ---- Precision@k for k = 1 … 10 -------------------------  # CHANGED
        for k_prec in ks:                                 # CHANGED
            k_eff = min(k_prec, total_pos)                          # CHANGED
            hits  = sum(labels_sorted[:k_prec])                     # CHANGED
            prec_lists[k_prec].append(hits / k_eff)                 # CHANGED



        aps_full.append(average_precision_at_k(labels_sorted))
        for k in ks:
            aps_k[k].append(average_precision_at_k(labels_sorted, k))

  
    out = {                                                      # CHANGED
            "ap": float(np.mean(aps_full)) if aps_full else 0.0,     # CHANGED
            "ap_all": aps_full                                       # CHANGED
        }  
    
    for k_prec in ks:                                  # CHANGED
        out[f"prec@{k_prec}"]      = float(np.mean(prec_lists[k_prec])) if prec_lists[k_prec] else 0.0  # CHANGED
        out[f"prec@{k_prec}_all"]  = prec_lists[k_prec]                                                                        # CHANGED


    for k in ks:
        out[f"ap@{k}"] = float(np.mean(aps_k[k])) if aps_k[k] else 0.0
        out[f"ap@{k}_all"] = aps_k[k]

    return out


def add_node_level_ap(df: pd.DataFrame, ks=(1,2, 3, 4, 5,6, 7, 8, 9, 10)):
    """
    Enrich *df* (in-place) with 'ap', 'ap@3', 'ap@5', …

    * df must already contain 'probability', 'label', 'meta'.
    * Returns the same dataframe for convenience.
    """
  

    ap_cols = {col: [] for col in                                  # CHANGED
               ["ap"] + [f"ap@{k}" for k in ks] + ["ap_all"] +     # CHANGED
               [f"ap@{k}_all" for k in ks] +                       # CHANGED
               [f"prec@{k}" for k in ks] +               # CHANGED
               [f"prec@{k}_all" for k in ks]}            # CHANGED

    assert len(df["probability"]) == len(df["label"]) == len(df["meta"]), "Length mismatch in DataFrame columns"

    for probs, labs, metas, approach in zip(df["probability"],
                                  df["label"],
                                  df["meta"],
                                  df["approach"]):
        if (approach=="Random"):
            print("he")
        #we compute the map shores per graph and add them
        metrics = map_per_node(probs, labs, metas, ks)
        for col, val in metrics.items():
            ap_cols[col].append(val)
        

    for col, vals in ap_cols.items():
        df[col] = vals  #changed

    summary_cols = ["key", "approach", "MAP@3", "MAP@5", "MAP@10", "MAP (full)", "prec@5"]
    # Rename columns in df for clarity
    df.rename(columns={
        "ap": "MAP (full)",
        "ap@3": "MAP@3",
        "ap@5": "MAP@5",
        "ap@10": "MAP@10"
    }, inplace=True)
    missing = [c for c in summary_cols if c not in df.columns]
    if missing:
        print(f"- skipped per-approach table (missing columns: {missing})")
    else:
        nice = (
            df[summary_cols]
            .sort_values("MAP@5", ascending=False)
            .reset_index(drop=True)
        )

        #print(nice.to_string(index=False, float_format="%.3f"))

        print("\n--- per approach ---")
        print(
            nice.groupby("approach")[["MAP@3", "MAP@5", "MAP@10",
                                    "MAP (full)", "prec@5" ]]         #CHANGED
            .mean()
            .round(3)
        )

    return df