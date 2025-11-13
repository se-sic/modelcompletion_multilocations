import os
import json
from matplotlib import pyplot as plt
import pandas as pd

def compute_average_metric_per_dataset(df_all, metric_name):
    """
    Compute the average of a given metric:
    1. First averaged per dataset within each source.
    2. Then averaged across datasets per source.
    """
    if metric_name not in df_all.columns:
        raise ValueError(f"Metric '{metric_name}' not found in DataFrame.")

    df_avg_dataset = (
        df_all.groupby(["source", "dataset"])[metric_name]
        .mean()
        .reset_index()
    )

    df_avg_source = (
        df_avg_dataset.groupby("source")[metric_name]
        .mean()
        .reset_index()
        .rename(columns={metric_name: f"avg_{metric_name}"})
    )

    return df_avg_source

def compute_average_metric(df_all, metric_name, group_by=["source"]):
    """
    Compute the average of a given metric grouped by specified columns.
    Example: group_by=["source", "dataset"]
    """
    if metric_name not in df_all.columns:
        raise ValueError(f"Metric '{metric_name}' not found in DataFrame.")

    avg = (
        df_all
        .groupby(group_by)[metric_name]
        .mean()
        .reset_index()
        .rename(columns={metric_name: f"avg_{metric_name}"})
    )
    return avg




def plot_all_metrics_over_iterations(df_all):
    """
    Plot average metric value over iterations per source for all numeric metrics.
    """
    # Identify numeric metrics (excluding metadata columns)
    exclude_cols = {"source", "dataset", "starting_node", "iteration", "path"}
    metric_names = [c for c in df_all.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_all[c])]

    for metric_name in metric_names:
        df_iter = (
            df_all.groupby(["source", "iteration"])[metric_name]
            .mean()
            .reset_index()
            .sort_values("iteration")
        )

        plt.figure(figsize=(8, 5))
        for source in df_iter["source"].unique():
            df_sub = df_iter[df_iter["source"] == source]
            plt.plot(df_sub["iteration"], df_sub[metric_name], marker="o", label=source)

        plt.title(f"{metric_name} over iterations")
        plt.xlabel("Iteration")
        plt.ylabel(metric_name)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
def load_all_metrics(base_path):
    """
    Recursively load all metrics files from a given top-level folder.
    Expected structure:
    base_path/
        dataset_name/
            starting_node_X/
                results_metricsN.json
    """
    all_records = []

    for dataset in os.listdir(base_path):
        dataset_path = os.path.join(base_path, dataset)
        if not os.path.isdir(dataset_path):
            continue

        for starting_node in os.listdir(dataset_path):
            node_path = os.path.join(dataset_path, starting_node)
            if not os.path.isdir(node_path):
                continue

            for file in os.listdir(node_path):
                if file.startswith("results_metrics") and file.endswith(".json"):
                    file_path = os.path.join(node_path, file)
                    try:
                        with open(file_path, "r") as f:
                            data = json.load(f)
                        data.update({
                            "dataset": dataset,
                            "starting_node": int(starting_node.split("_")[-1]),
                            "iteration": int(file.split("results_metrics")[-1].split(".json")[0]),
                            "path": file_path
                        })
                        all_records.append(data)
                    except Exception as e:
                        print(f"⚠️ Could not load {file_path}: {e}")

    return pd.DataFrame(all_records)

from scipy.stats import wilcoxon
def wilcoxon_for(metric):
    # ensure numeric float (not bool)
    df_float = df_all.copy()
    df_float[metric] = df_float[metric].astype(float)

    # 1) mean per dataset per source
    df_ds = (
        df_float.groupby(["source", "dataset"])[metric]
        .mean()
        .reset_index()
    )

    # 2) pivot to paired samples
    pivot = df_ds.pivot(index="dataset", columns="source", values=metric).dropna()

    # 3) paired Wilcoxon test
    x = pivot["NextFocus"].astype(float)
    y = pivot["Tinnes et al"].astype(float)

    stat, p = wilcoxon(x, y, zero_method="wilcox")

    print(f"{metric}: W={stat:.3f}, p={p:.4f}, NextFocus_mean={x.mean():.3f}, Tinnes_mean={y.mean():.3f}")


# === Example usage ===
path_tinnes = "../resultsmajorrevision/groundTruths/ours_final_chatgpt4"
path_ours   = "../resultsmajorrevision/groundTruths/ours10_all"

df_tinnes = load_all_metrics(path_tinnes)
df_ours   = load_all_metrics(path_ours)

# Combine for joint analysis (add source label)
df_tinnes["source"] = "Tinnes et al."
df_ours["source"]   = "NextFocus"
df_all = pd.concat([df_tinnes, df_ours], ignore_index=True)


# === Add hierarchy constraints here ===
# CHANGED
df_all["change_correct"] = df_all["change_correct"] & df_all["structure_correct"] #& df_all ["correct_next_focus"]   # CHANGED
df_all["type_structure"] = df_all["type_structure"] & df_all["change_correct"] & df_all["structure_correct"]#& df_all ["correct_next_focus"]  # CHANGED
# ===

#df_all = df_all[df_all["iteration"] >0 ] 

print(f"Loaded {len(df_all)} metric files in total.")
print(df_all.head())

avg_next_focus = compute_average_metric_per_dataset(df_all, "correct_next_focus")
print(avg_next_focus)
wilcoxon_for("correct_next_focus")

avg_next_focus_format_correct= compute_average_metric_per_dataset(df_all, "format_correct")
print(avg_next_focus_format_correct)
wilcoxon_for("format_correct")

avg_next_focus_structure_correct= compute_average_metric_per_dataset(df_all, "structure_correct")
print(avg_next_focus_structure_correct)
wilcoxon_for("structure_correct")

avg_next_focus_change_correct = compute_average_metric_per_dataset(df_all, "change_correct")
print(avg_next_focus_change_correct)
wilcoxon_for("change_correct")

avg_next_focus_type_structure = compute_average_metric_per_dataset(df_all, "type_structure")
print(avg_next_focus_type_structure)
wilcoxon_for("type_structure")
#plot_all_metrics_over_iterations(df_all)

