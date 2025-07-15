
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Get the grandparent directory (parent of the parent)
grandparent_dir = os.path.dirname(parent_dir)
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

# Add the grandparent directory to sys.path
sys.path.insert(0, grandparent_dir)


from experiments.crosscutting_changes.HELPER_final_eval import add_node_level_ap, filter_keys_with_all_approaches, load_final_results, prune_leaf_edges

from experiments.crosscutting_changes.final_eval_graph_radius import filter_radius
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Get the grandparent directory (parent of the parent)
grandparent_dir = os.path.dirname(parent_dir)
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

# Add the grandparent directory to sys.path
sys.path.insert(0, grandparent_dir)

save_output_path = ""
overall_order= ["Random", "Historical", "Semantics", "NeuralNetwork"]
def plot_comparison_box(df):
    df_exploded = df.explode("precision_all").copy()
    df_exploded["precision_all"] = pd.to_numeric(df_exploded["precision_all"], errors="coerce")

    plt.figure(figsize=(10, 4))
    ax = sns.boxplot(
        data=df_exploded,
        x="approach",
        y="precision_all",
        order=overall_order,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black"},
        fliersize=2,
        linewidth=1
    )

    plt.ylabel("Top-5 Precision", fontsize=12)
    plt.xlabel("Approach", fontsize=12)
    plt.title("Top-5 Precision Distribution (Box Plot)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_output_path, bbox_inches="tight", dpi=300)
    plt.show()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_comparison_violin_all_overlay(df, metric_tpl="prec@{k}_all", ks=range(1, 11)):  # CHANGED
    # Create a long-form DataFrame manually  
    records = {}
    for k in ks:  
        col = metric_tpl.format(k=k)  
        for _, row in df.iterrows(): 
            approach = row["approach"] 
            values = row[col] if isinstance(row[col], list) else [row[col]]  
            for v in values: 
                if approach not in records: 
                    records[approach] = {} 
                if k not in records[approach]:  
                    records[approach][k] = []  
                for v in values:
                    records[approach][k].append(v) 

     # Flatten into long_df
    rows = []  
    for approach, k_dict in records.items(): 
        for k, vals in k_dict.items():  
            for v in vals:
                rows.append({"approach": approach, "k": k, "value": v})  
    long_df = pd.DataFrame(rows) 



    plt.rcParams.update({
    'font.size': 15,
    'axes.labelsize': 15,      # x/y axis label size
    'xtick.labelsize': 15,     # x-axis tick label size
    'ytick.labelsize': 15,     # y-axis tick label size
    'legend.fontsize': 15,     # legend text size
    'axes.titlesize': 15       # title size
})
    plt.figure(figsize=(13, 5))
    ax = plt.gca()
   
   
    palette = sns.color_palette("tab10", n_colors=len(ks))  
    for idx, k in enumerate(ks):  
        sns.violinplot(
            data=long_df.query("k == @k"),  
            x="approach",
            y="value",
            order=overall_order,
            cut=0,
            inner=None,
            linewidth=1.2,
            bw=0.6,      
            fill=False,
            color=palette[idx], 
        )

        # ----- overlay group-mean dots -----------------------------------
        group = {}
        for r in long_df.query("k == @k").to_dict("records"):
            group.setdefault(r["approach"], []).append(r["value"])
        for i, approach in enumerate(overall_order):
            vals = group.get(approach, [])
            m = np.mean(vals) if vals else np.nan           
            if not np.isnan(m):                              
                ax.scatter(i, m, s=50, alpha=0.5,zorder=10,
                           color=palette[idx],
                            label=f"k={k}" if i == 0 else "")  # CHANGED)  
                print(f"Mean for approach={approach}, k={k}: {m:.4f}")  # CHANGED
     # ---------- legend outside -------------------------------------------
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
            title="Precision@k",    
              bbox_to_anchor=(1.02, 1), loc="upper left",
              borderaxespad=0,
              frameon=True,     
              fancybox=False) 

    xtick_labels = [label.get_text() for label in ax.get_xticklabels()]
    xtick_labels = ["NextFocus" if lbl == "NeuralNetwork" else lbl for lbl in xtick_labels]
    ax.set_xticklabels(xtick_labels)  # CHANGED

    ax.set_xlabel("")  
    ax.set_ylabel("Probability Density")                       
    plt.tight_layout(rect=[0, 0, 0.85, 1])   

    for approach in overall_order:
        all_vals = long_df.query("approach == @approach")["value"].values
        if len(all_vals) > 0:
            overall_mean = np.mean(all_vals)
            std_dev = np.std(all_vals)
            zero_count = np.sum(all_vals == 0)
            nonzero_count = np.sum(all_vals != 0)
            print(f"Approach: {approach}")
            print(f"  Mean: {overall_mean:.4f}")
            print(f"  Std Dev: {std_dev:.4f}")
            print(f"  Zero count: {zero_count} ({zero_count / len(all_vals):.1%})")
            print(f"  Non-zero count: {nonzero_count} ({nonzero_count / len(all_vals):.1%})")



    output_path = os.path.expanduser("~/Downloads/violin_plot.pdf")  # expands ~ to full path
    plt.savefig(output_path, format="pdf", bbox_inches="tight")

    plt.show()  





def plot_nn_vs_other(df, other_approach):
    """Plot NeuralNetwork precision vs. another approach's precision."""
    # Remove entries with invalid precision
    df = df[df["precision_avg"] != -1]

    # Aggregate by key + approach to avoid duplicates
    df_grouped = df.groupby(["key", "approach"])["precision_avg"].mean().reset_index()

    # Pivot for comparison
    pivot_df = df_grouped.pivot(index="key", columns="approach", values="precision_avg")

    # Drop rows where either value is missing
    pivot_df = pivot_df.dropna(subset=["NeuralNetwork", other_approach])

    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(pivot_df["NeuralNetwork"], pivot_df[other_approach], alpha=0.7)
    plt.plot([0, 1], [0, 1], "--", color="gray")  # y = x line
    plt.xlabel("NeuralNetwork Precision")
    plt.ylabel(f"{other_approach} Precision")
    plt.title(f"NeuralNetwork vs {other_approach}")
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
def plot_box_like_hist_per_approach(df):
    df_exploded = df.explode("precision_all").copy()
    df_exploded["precision_all"] = pd.to_numeric(df_exploded["precision_all"], errors="coerce")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

    for i, approach in enumerate(overall_order):
        ax = axes[i]
        subset = df_exploded[df_exploded["approach"] == approach]
        counts = subset["precision_all"].value_counts().sort_index()

        ax.bar(counts.index, counts.values, width=0.1)
        ax.set_title(approach, fontsize=14)
        ax.set_xlabel("Top-5 Precision", fontsize=12)
        if i == 0:
            ax.set_ylabel("Count", fontsize=12)
        ax.set_xlim(0, 1)
        ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    plt.suptitle("Distribution of Precision Values per Approach", fontsize=16)
    plt.tight_layout()
    plt.show()




def assert_equal_lengths(df):
    """
    Ensure probability / label / meta lists are the same length for every row.
    Raise an AssertionError that names the first offending approach + key.
    """
    bad_rows = []

    for idx, row in df.iterrows():
        n_prob  = len(row["probability"])
        n_lab   = len(row["label"])
        n_meta  = len(row["meta"])
        if not (n_prob == n_lab == n_meta):
            bad_rows.append(
                {
                    "index": idx,
                    "key": row.get("key", "<no-key>"),
                    "approach": row.get("approach", "<no-approach>"),
                    "prob_len": n_prob,
                    "lab_len": n_lab,
                    "meta_len": n_meta,
                }
            )

    if bad_rows:                                        # at least one mismatch
        first = bad_rows[0]
        msg = (
            f"Length mismatch in row {first['index']} "
            f"(approach='{first['approach']}', key='{first['key']}'): "
            f"prob={first['prob_len']}, lab={first['lab_len']}, meta={first['meta_len']}"
        )
        # optional: show all offending rows
        print("⚠  All mismatching rows:")
        for r in bad_rows:
            print(
                f"  idx={r['index']:>4}  "
                f"approach={r['approach']:<12}  "
                f"prob={r['prob_len']:>5}  lab={r['lab_len']:>5}  meta={r['meta_len']:>5}"
            )
        raise AssertionError(msg)

# ----------------------------------------------------
# call it right after pruning & before further metrics

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D                      # NEW


# stats_wilcoxon_precision.py
import itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def _collect_precisions(df: pd.DataFrame, ks):
    """Return dict[(approach, k)] -> list[float]"""
    data = {}
    for k in ks:
        col = f"prec@{k}_all"
        if col not in df.columns:
            raise KeyError(f"Missing column {col!r}")
        for appr, values in zip(df["approach"], df[col]):
            data.setdefault((appr, k), []).extend(values)
    return data


def _wilcoxon_matrix(data, approaches, k, alpha):
    """Return DataFrame of p-values, star if p < alpha."""
    mat = pd.DataFrame(index=approaches, columns=approaches, dtype=object)
    for a, b in itertools.combinations(approaches, 2):
        x, y = data[(a, k)], data[(b, k)]
        p = wilcoxon(x, y, zero_method="zsplit", alternative="two-sided").pvalue
        mark = "*" if p < alpha else ""
        mat.loc[a, b] = mat.loc[b, a] = f"{p:.3g}{mark}"
    np.fill_diagonal(mat.values, "-")
    return mat

from scipy.stats import wilcoxon
import numpy as np
from scipy.stats import mannwhitneyu


def neuralnet_vs_others_manwid(df, target, ks=range(1, 11), alpha=0.01):
    """
    One-sided Wilcoxon test: is NeuralNet significantly better than each other approach?
    For each precision@k, prints: p-value and direction.
    """
      # must match your df["approach"] name exactly
    other_approaches = sorted(set(df["approach"]) - {target})
    results = [] 

    for k in ks:
        col = f"prec@{k}_all"
        if col not in df.columns:
            continue

        # collect per-graph precision@k
        nn_values = []
        buckets = {}
        for appr, vals in zip(df["approach"], df[col]):
            if appr == target:
                nn_values.extend(vals)
            else:
                buckets.setdefault(appr, []).extend(vals)

        for other, y_vals in buckets.items():
            x_vals = nn_values
            u, p = mannwhitneyu(x_vals, y_vals, alternative="greater")
            signif = "significantly better" if p < alpha else "not significantly better"

            status = "yes" if  p < alpha else "no"  # CHANGED: mark significance

            results.append({  # CHANGED: store result
                "k": k,
                "vs": other,
                "p-value": f"{p:.3g}",
                "significant": status
            })

            print(f"Precision@{k}: {target} is {signif} than {other} (p = {p:.3g})")

    # CHANGED: build LaTeX tables
    df_out = pd.DataFrame(results)
    latex = df_out.pivot(index="k", columns="vs", values="p-value").fillna("-")  # CHANGED
    sigs = df_out.pivot(index="k", columns="vs", values="significant").fillna("")  # CHANGED

    print("\nLaTeX Table (p-values):")  # CHANGED
    print(latex.to_latex(escape=False, caption=f"Mann–Whitney U test: {target} vs. others", label="tab:nn_vs_others_pvalues"))  # CHANGED

    print("\nLaTeX Table (✓ = significant):")  # CHANGED
    print(sigs.to_latex(escape=False, caption="Significance of differences (✓ = p < 0.05)", label="tab:nn_vs_others_significance"))  # CHANGED


def test_difference_average(df, metric_tpl, ks=range(1, 11)):
    from scipy.stats import shapiro
    from scipy.stats import kruskal
    def check_normal(): 
        results = {}
        for k in ks:
            col = metric_tpl.format(k=k)
            data = df[col].dropna()
            if len(data) < 3:
                print(f"Not enough data for {col} (n={len(data)})")
                continue
            stat, pval = shapiro(data)
            print(f"Shapiro-Wilk test for '{col}': p = {pval:.3g}")
            results[k] = (stat, pval)
        return results
    
    #check_normal()

    #def kruskal_test_per_k(df, metric_tpl, ks=range(1, 11)):
    results = {}
    for k in ks:
        col = metric_tpl.format(k=k)
        # df[col] is list-like — group and flatten correctly
        grouped = df.groupby("key")[col].apply(
            lambda series: sum(series, [])  # flatten list of lists
        )
        groups = [v for v in grouped if len(v) > 1]  # only use groups with enough data

        if len(groups) < 2:
            print(f"Not enough valid project groups for {col}")
            continue

        stat, pval = kruskal(*groups)
        print(f"Kruskal-Wallis test for '{col}': p = {pval:.3g}")
        results[k] = (stat, pval)
    return results


import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np

import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

if __name__ == "__main__":
    # Load data
    file_path = "../Final_eval/output_ONLYONE4/"
 
    df = load_final_results(file_path)
    df = filter_keys_with_all_approaches(df)
    df = prune_leaf_edges(df) 
    # show first 10 rows with their list lengths
   
    df = add_node_level_ap(df)


    #never used
    #plot_nn_vs_other(df, "Historical")
   # neuralnet_vs_others_manwid(df, target=  "NeuralNetwork")     
    #neuralnet_vs_others_manwid(df, target=  "Historical")     
   # n#euralnet_vs_others_manwid(df, target=  "Random")          
    #neuralnet_vs_others_manwid(df, target=  "Semantics")         

    #important
    #plot_comparison_violin_all(df, "prec@2_all")
    #df = filter_radius(
     #       df,
      #      min_dist=2
       # )
    #plot_comparison_violin_all_overlay(df, metric_tpl="prec@{k}_all")
    plot_comparison_violin_all_overlay(df, metric_tpl="prec@{k}")
  #  test_difference_average(df, metric_tpl="prec@{k}_all")
    # erstellt wir box plots, x axis: precision, y: how many times 
    #plot_box_like_hist_per_approach(df)


   

