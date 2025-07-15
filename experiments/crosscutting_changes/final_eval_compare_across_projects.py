import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Get the grandparent directory (parent of the parent)
grandparent_dir = os.path.dirname(parent_dir)
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

# Add the grandparent directory to sys.path
sys.path.insert(0, grandparent_dir)

from experiments.crosscutting_changes.HELPER_final_eval import load_final_results
import pandas as pd
from experiments.crosscutting_changes.final_comparison import overall_order


import matplotlib.pyplot as plt
import seaborn as sns


def plot_comparison(df):
    """Plot precision comparisons between approaches across different keys."""
    plt.figure(figsize=(12, 6))
   # sns.boxplot(data=df, x="approach", y="precision_avg")


    sns.boxplot(
    data=df,
    x="approach",
    y="precision_avg",
    order=overall_order
)
    plt.xticks(rotation=45)
    plt.xlabel("Approach")
    plt.ylabel("Precision Average")
    plt.title("Comparison of Precision Across Approaches")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def sort_keys_by_approach_count(df):
    """Sort keys based on how many unique approaches have results."""
    key_counts = df.groupby("key")["approach"].nunique().reset_index()

    expected_approaches = set(df["approach"].unique())
    grouped = df.groupby("key")["approach"].unique()
    for key, approaches_present in grouped.items():
        missing = expected_approaches - set(approaches_present)
        if missing:
            print(f"Key: {key} is missing approaches: {sorted(missing)}")

   # Keep only keys that have at least 3 approaches
    valid_keys = key_counts[key_counts["approach"] >= 4]["key"]
    df = df[df["key"].isin(valid_keys)]

    #TODO this is new why?
    df["precision_avg"] = pd.to_numeric(df["precision_avg"], errors="coerce")

    df_agg = df.groupby(["key", "approach"], as_index=False)["precision_avg"].mean()


    # Pivot to create a table where each approach is a column
    pivot_df = df_agg.pivot(index="key", columns="approach", values="precision_avg").reset_index()

    # Sorting order based on specified approaches
    sort_order = overall_order

    # Sort first by neural_network, then semantics_results, etc.
    pivot_df = pivot_df.sort_values(by=sort_order, ascending=False, na_position='last')

    # Apply sorted key order to the original dataframe
    sorted_keys = pivot_df["key"].tolist()
    df["key"] = pd.Categorical(df["key"], categories=sorted_keys, ordered=True)
    return df.sort_values("key")


def plot_key_comparison(df):
    df = sort_keys_by_approach_count(df)  # Ensure keys are sorted before plotting

    """Plot how each approach performs on each dataset project key."""
    plt.figure(figsize=(14, 7))
    sns.barplot(data=df, x="key", y="precision_avg", hue="approach")
    plt.xticks(rotation=90)
    plt.xlabel("Dataset Project Key")
    plt.ylabel("Precision Average")
    plt.title("Precision Performance per Key Across Approaches")
    plt.legend(title="Approach")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


if __name__ == "__main__":
    # Load data
    file_path = "../Final_eval/output_ONLYONE/"
    #file_path="../output_dataset_label/eval/nn/"
    df = load_final_results(file_path)
 
    # TODO größere schrift 
    # Aufrufen:
    #plot_nn_vs_other(df, "Semantics")
    plot_key_comparison(df)
   
   
