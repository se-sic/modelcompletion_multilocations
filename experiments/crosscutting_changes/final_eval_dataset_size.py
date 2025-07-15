import os
import sys
import seaborn as sns  # CHANGED
import math

from sklearn.isotonic import spearmanr
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Get the grandparent directory (parent of the parent)
grandparent_dir = os.path.dirname(parent_dir)
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

# Add the grandparent directory to sys.path
sys.path.insert(0, grandparent_dir)
from itertools import cycle  # CHANGED
line_styles = cycle(['--', '-.', ':', (0, (3, 1, 1, 1))])  # CHANGED


from experiments.crosscutting_changes.HELPER_final_eval import add_node_level_ap, load_final_results, prune_leaf_edges


import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import numpy as np

#index 1 → total (the raw dataset size)
#index 2 → true (the “label =True” count)


data = [
    ("modeling.mmt.qvtd!!plugins_org.eclipse.qvt_model_ecore_QVT.ecore", 46960, 29471),
    ("modeling.mmt.qvtd!!plugins_org.eclipse.qvtd.pivot.qvtbase_model_QVTbase.ecore", 263367, 31279),
    ("modeling.gmp.gmf-tooling!!plugins_org.eclipse.gmf.map_models_gmfmap.ecore", 33341, 6933),
    ("modeling.emf.emf!!plugins_org.eclipse.emf.codegen.ecore_model_GenModel.ecore", 66900, 3530),
    ("technology.stem!!core_org.eclipse.stem.core_model_model.ecore", 2700, 444),
    ("technology.cbi!!org.eclipse.b3.aggregator.legacy_models_aggregator_0.9.0.ecore", 6132, 210),
    ("technology.cbi!!org.eclipse.b3.aggregator.legacy_models_aggregator_1.1.0.ecore", 27035, 2429),
    ("modeling.mdt.papyrus!!plugins_infra_core_org.eclipse.papyrus.infra.core.architecture_model_Architecture.ecore", 92, 2),
    ("modeling.emft.edapt!!plugins_org.eclipse.emf.edapt.declaration_model_declaration.ecore", 5902, 1262),
    ("modeling.mmt.qvtd!!plugins_org.eclipse.qvtd.pivot.qvttemplate_model_QVTtemplate.ecore", 56, 4),
    ("modeling.eef!!plugins_org.eclipse.emf.eef.modelingbot_model_mbot.ecore", 4878, 430),
    ("technology.stem!!core_org.eclipse.stem.core_model_graph.ecore", 3830, 102),
    ("modeling.mmt.qvtd!!plugins_org.eclipse.qvtd.xtext.qvtrelation_model_QVTrelationCS.ecore", 9145, 1242),
    ("technology.stem!!models_foodproduction_org.eclipse.stem.foodproduction_model_foodproduction.ecore", 130, 41),
    ("tools.buckminster!!org.eclipse.buckminster.rmap_model_rmap.ecore", 123725, 32746),
    ("modeling.mmt.qvt-oml!!plugins_org.eclipse.m2m.qvt.oml_model_QVTOperational.ecore", 19326340, 844047),
    ("technology.stem!!core_org.eclipse.stem.core_model_scenario.ecore", 27, 4),
    ("eclipse.e4!!bundles_org.eclipse.e4.ui.model.workbench_model_UIElements.ecore", 412937, 180948),
    ("modeling.mmt.qvtd!!plugins_org.eclipse.qvt_model_ecore_FlatQVT.ecore", 152346, 26684),
    ("modeling.emft.emf-client!!bundles_org.eclipse.emf.ecp.view.custom.model_model_custom.ecore", 50, 39),
    ("modeling.mmt.qvtd!!plugins_org.eclipse.qvtd.pivot.qvtrelation_model_QVTrelation.ecore", 486260, 44009),
    ("modeling.mdt.uml2!!plugins_org.eclipse.uml2.uml_model_CMOF.ecore", 5360537, 2154330),
    ("technology.cbi!!org.eclipse.b3.backend_model_B3Backend.ecore", 585607, 24890),
    ("modeling.mmt.qvtd!!plugins_org.eclipse.qvtd.pivot.qvtimperative_model_QVTimperative.ecore", 92655, 32666),
    ("technology.cbi!!org.eclipse.b3.build_model_B3Build.ecore", 313432, 27334),
    ("modeling.mmt.qvtd!!plugins_org.eclipse.qvt_model_ecore_ImperativeOCL.ecore", 9961, 718),
    ("modeling.mmt.atl!!deprecated_org.atl.eclipse.engine_src_org_atl_eclipse_engine_resources_ATL-0.2.ecore", 21897, 6694),
    ("modeling.emft.edapt!!plugins_org.eclipse.emf.edapt.history_model_history.ecore", 8979, 663),
    ("modeling.emf.emf!!plugins_org.eclipse.emf.ecore_model_XMLType.ecore", 9049, 426),
    ("modeling.emf.emf!!plugins_org.eclipse.emf.ecore.change_model_Change.ecore", 725, 92),
    ("modeling.mdt.ocl!!plugins_org.eclipse.ocl.pivot_model_Lookup.ecore", 96, 18),
    ("technology.cbi!!org.eclipse.b3.aggregator_model_Aggregator.ecore", 453997, 89599),
    ("modeling.gmp.gmf-tooling!!plugins_org.eclipse.gmf.tooldef_models_tooldef.ecore", 246, 3),
    # newly added
    ("modeling.mmt.qvtd!!plugins_org.eclipse.qvtd.pivot.qvtcore_model_QVTcore.ecore", 7716, 4731),
    ("modeling.mdt.bpmn2!!org.eclipse.bpmn2_model_BPMN20.ecore", 828092, 10585)
]

#Purpose: turn your long data list of tuples into a dictionary that maps
#file_path → some numeric column you choose.
def build_count_lookup(data_tuples, field ) -> dict[str, int]:
  
    field = field.lower()
    if field not in {"total", "true", "pct"}:
        raise ValueError("field must be 'total', 'true', or 'pct'")

    lookup = {}
    for path, total, true_cnt in data_tuples:
        if field == "total":
            lookup[path] = total
        elif field == "true":
            lookup[path] = true_cnt
        else:                     # field == "pct"
            lookup[path] = true_cnt / total if total else float("nan")

    return lookup


############## PLOTTING ONLY####################

def plot_scatter_with_reg(tmp: pd.DataFrame, log_x: bool = True, min_pts: int = 2):
  
    approaches = tmp["approach"].unique()  # CHANGED
    approach_palette = {a: c for a, c in zip(approaches, sns.color_palette("tab10", len(approaches)))}  # CHANGED
    plt.rcParams.update({
    'font.size': 15,
    'axes.labelsize': 15,      # x/y axis label size
    'xtick.labelsize': 15,     # x-axis tick label size
    'ytick.labelsize': 15,     # y-axis tick label size
    'legend.fontsize': 15,     # legend text size
    'axes.titlesize': 15       # title size
})
    plt.figure(figsize=(11, 5))
   

    # collect overall data for global correlation if you like
    all_x, all_y = [], []

    for approach, grp in tmp.groupby("approach"): 

       
        g = grp[["size_metric", "precision"]].replace([np.inf, -np.inf], np.nan).dropna()
        if g.empty:
            continue

        x = g["size_metric"].values
        y = g["precision"].values
    
        color = approach_palette[approach]

       # style = next(line_styles)

        if approach == "NeuralNetwork":  #  Only show scatter for NN
            plt.scatter(x, y, alpha=0.5,s=50, color=color)

        if len(x) >= min_pts and np.unique(x).size >= 2:
            x_fit = np.log10(x).reshape(-1, 1) if log_x else x.reshape(-1, 1)
            model = LinearRegression().fit(x_fit, y)
            x_line = np.linspace(x.min(), x.max(), 150)
            x_line_fit = np.log10(x_line).reshape(-1, 1) if log_x else x_line.reshape(-1, 1)
            y_pred = model.predict(x_line_fit)
            plt.plot(x_line, y_pred, linestyle="--", linewidth=1.5, color=color)





    # global correlation (entire pooled sample)
    if all_x:
        xx = np.concatenate(all_x)
        yy = np.concatenate(all_y)
        if len(xx) >= min_pts and np.unique(xx).size >= 2:
            r_all, _ = pearsonr(xx, yy)
            print(f"Overall          : r = {r_all:.2f}  (n={len(xx)})")

    # cosmetics
    if log_x:
        plt.xscale("log")
        plt.xlabel("Number of data point in the train test")
    else:
        plt.xlabel("Number of data point in the train test")


    handles = []
    labels = []
    for approach, base_color in approach_palette.items():      # CHANGED
                                     # CHANGED
        handles.append(plt.Line2D([0], [0], color=base_color, linestyle='-', linewidth=2))  # CHANGED
        labels.append("NextFocus" if approach == "NeuralNetwork" else approach)  #

    plt.legend(handles, labels,            
           bbox_to_anchor=(1.02, 1.0), loc="upper left",    # ✅ CHANGED: move to the right
           borderaxespad=0.0,
           frameon=True,
           fancybox=False)

    xticks = plt.xticks()[0]
    xticklabels = [label.get_text() for label in plt.gca().get_xticklabels()]
    new_labels = ["NextFocus" if lbl == "NeuralNetwork" else lbl for lbl in xticklabels]
    plt.gca().set_xticklabels(new_labels)  # CHANGED

    ymin, ymax = plt.ylim()
    plt.ylim(bottom=-0.01 * (ymax - ymin))  # ✅ allow small margin below 0 
    plt.ylabel("Average Top-k Precision per Project")
    #plt.title(f"{precision_col} vs. # true labels")
    plt.grid(True, which="both", linestyle="--", linewidth=0.3)
    plt.subplots_adjust(right=0.75) 
    #plt.tight_layout(rect=(0.3, 0, 1, 1)) 
    plt.savefig(os.path.expanduser("~/Downloads/plot_datasets.pdf"), format="pdf", bbox_inches="tight")
 
    plt.show()


##############PLOTTING ENDE ##############

#COMPARISON TO TEST SWET
# wir plotten precision @5, vs wie viele changed items wir im test set haben, 
def compare_testset_size(df: pd.DataFrame, top_k: int = 5):
    df = df.copy()
    df["group_size"] = df["precision_all"].apply(lambda x: len(x) if isinstance(x, list) else 0)

    plt.figure(figsize=(8, 5))
    df = df[df["approach"] == "Random"]
    # Print one line per row
    #  print("Approach;Key;Precision_avg;Size")
    for _, r in df.iterrows():
        precisions = []
        for k in range(1, 11):
            key = f'prec@{k}'
            precisions.append(r[key])
        avg_precision = sum(precisions) / len(precisions) 
        size = len(r['prec@3_all']) if isinstance(r['prec@3_all'], list) else 0
        print(f"{r['approach']};{r['key']};{avg_precision:.3f};{size}")
    
    for name, group in df.groupby("approach"):
        # Scatter plot
        plt.scatter(
            group["group_size"],
            group["precision_avg"],
            alpha=0.5,
            label=name
        )

        # Linear regression line
        x = group["group_size"].values.reshape(-1, 1)
        y = group["precision_avg"].values
        if len(x) >= 2:
            model = LinearRegression().fit(x, y)
            x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
            y_pred = model.predict(x_range)
            plt.plot(x_range, y_pred, linestyle='--', label=f"{name} (trend)")

    plt.xlabel("Group size (len(precision_all))")
    plt.ylabel("Precision@k")
    plt.title("Precision@k vs. Group size (with regression)")
    plt.grid(True)
    plt.xlim(0, 200)
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Pearson correlation on raw data
    corr, _ = pearsonr(df["group_size"], df["precision_avg"])
    print(f"Pearson correlation (overall): {corr:.2f}")




#number of data points, plotted vs precision 
def compare_trainset_size(df: pd.DataFrame, comparison ,
                             metric) -> None:
     # --- helper: build a lookup "file → #true labels" from your data list ------------
    # or total "total", TODO with true we need to think if it makes more sentsne to take /true/ttoal
    size_lookup = build_count_lookup(data, field=comparison)
    #"total", "true", "pct"
    # Step 1: compute candidate size
    df["candidate_size"] = df["precision_all"].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # Step 2: compute dynamic k per row
    alpha = 0.01  # exaggerated for testing
    df["k_dynamic"] = (df["candidate_size"] * alpha).clip(lower=1).round().astype(int)

    # Step 3: create new df_x filtered per key and per k
    rows = []
    for key in df["key"].unique():
        df_x = df[df["key"] == key].copy()
        k = df_x["k_dynamic"].iloc[0]

        df_x_ap = add_node_level_ap(df_x, ks=[k])  # <- keep only this k
        metric_col = metric.format(k=k)
        for _, row in df_x_ap.iterrows():
            rows.append({
                "key": row["key"],
                "approach": row["approach"],
                "k_dynamic": k,
                "precision@k_dynamic": row[metric_col]
            })

    tmp = pd.DataFrame(rows)
    metric= "precision@k_dynamic"
    


    tmp["size_metric"] = tmp["key"].map(size_lookup)

    tmp_long = tmp.rename(columns={metric: "precision"})  
    tmp_long = tmp_long[["size_metric", "approach", "key", "k_dynamic", "precision"]]  # CHANGED
    tmp_long = tmp_long.rename(columns={"k_dynamic": "k"})  # CHANGED


    plot_scatter_with_reg(tmp_long)  


    print("\nSpearman correlation between dataset size and precision:")
    for approach, grp in tmp_long.groupby("approach"):  # CHANGED
        valid = grp[["size_metric", "precision"]].dropna()
        if len(valid) >= 2:
            r, p = spearmanr(valid["size_metric"], valid["precision"])
            print(f"{approach:<15}: ρ = {r:.2f}, p = {p:.3f}, n = {len(valid)}")  # CHANGED
        else:
            print(f"{approach:<15}: not enough data")  # CHANGED










#computes the difference between semantics and nn with respect to top5 in depende on dataset size
#-> sometimes for less data NN is better 
def plot_performance_difference(df: pd.DataFrame):
    df = df.copy()
    df["group_size"] = df["precision_all"].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # Pivot to compare NeuralNetwork and Semantic directly
    pivot = df.pivot_table(index="key", columns="approach", values="precision_avg", aggfunc="mean")

    group_size_map = df.groupby("key")["group_size"].mean()
    pivot["group_size"] = pivot.index.map(group_size_map)
    pivot["diff"] = pivot["Semantics"] - pivot["NeuralNetwork"]

    plt.figure(figsize=(8, 5))
    plt.axhline(0, color='gray', linestyle='--')

    # Scatter plot colored by who wins
    plt.scatter(
        pivot["group_size"],
        pivot["diff"],
        c=(pivot["diff"] > 0),  # True: Semantic better
        cmap="bwr",
        alpha=0.6,
        label="Semantic - NeuralNetwork"
    )

    #for idx, row in pivot.iterrows():
    #    x = row["group_size"]
    #    y = row["diff"]
    #    color = "red" if y > 0 else "blue"
    #    plt.scatter(x, y, color=color, alpha=0.6)
    #    plt.text(x + 1, y, str(idx), fontsize=8, alpha=0.7)  # label with key

        

    plt.xlabel("Group size")
    plt.ylabel("Precision difference (Semantic - NeuralNetwork)")
    plt.title("Performance difference vs. Group size")
    plt.grid(True)
    plt.xlim(0, 200)
    plt.ylim(-1.05, 1.05)
    plt.tight_layout()
    plt.show()

    corr, _ = pearsonr(pivot["group_size"], pivot["diff"])
    print(f"Pearson correlation (group size vs. Semantic - NeuralNetwork): {corr:.2f}")






file_path = "../Final_eval/output_ONLYONE4"
#file_path="../output_dataset_label/eval/nn/"
df =load_final_results(file_path)
df = prune_leaf_edges(df) 
df = add_node_level_ap(df)
#compare either to "total" number,  "true" items, or percentage of true item of voerall "pct"
compare_trainset_size(df, comparison="true", metric="prec@{k}")
#äcompare_testset_size(df)
#plot_performance_difference(df)