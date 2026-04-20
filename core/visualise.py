import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.mlp import MLP
from optimisers.adam import Adam

# Load results
file_path = Path("experiments") / "exp1" /"exp1.json"

with file_path.open() as f:
    results = json.load(f)

df = pd.DataFrame(results)
df["accuracy"] = df["accuracy"].astype(float)


# 3D grid search visualization
fig = px.scatter_3d(
    df,
    x="hidden_size",
    y="lr",
    z="epochs",
    color="accuracy",
    color_continuous_scale="inferno",
    hover_data=["hidden_size", "lr", "epochs", "accuracy"]
)

fig.update_layout(
    title="Grid Search: Accuracy across Hidden Size, LR, Epochs"
)

fig.show()
plt.savefig()

# Heatmaps per hidden_size
for hs in sorted(df["hidden_size"].unique()):

    subset = df[df["hidden_size"] == hs].pivot_table(
        index="epochs",
        columns="lr",
        values="accuracy",
        aggfunc="max"
    )

    plt.figure(figsize=(10, 6))

    sns.heatmap(
        subset,
        annot=True,
        fmt=".2f",
        cmap="plasma",
        cbar_kws={"label": "Accuracy"}
    )

    plt.title(f"Accuracy Heatmap (hidden_size={hs})")
    plt.xlabel("Learning Rate")
    plt.ylabel("Epochs")

    plt.show()


# Accuracy vs Epochs (per learning rate)
for lr in sorted(df["lr"].unique()):

    plt.figure(figsize=(10, 6))

    subset = df[df["lr"] == lr]

    for hs in sorted(subset["hidden_size"].unique()):

        hs_subset = subset[subset["hidden_size"] == hs]
        hs_subset = hs_subset.sort_values(by="epochs")

        plt.plot(
            hs_subset["epochs"],
            hs_subset["accuracy"],
            marker="o",
            label=f"hidden_size={hs}"
        )

    plt.title(f"Accuracy vs Epochs (learning_rate={lr})")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()


# Aggregate heatmap (max accuracy across all hidden_sizes)
agg = df.groupby(["epochs", "lr"])["accuracy"].max().reset_index()

pivot = agg.pivot(
    index="epochs",
    columns="lr",
    values="accuracy"
)

plt.figure(figsize=(10, 6))

sns.heatmap(
    pivot,
    annot=True,
    fmt=".2f",
    cmap="plasma",
    cbar_kws={"label": "Max Accuracy"}
)

plt.title("Max Accuracy Heatmap across Hidden Sizes")
plt.xlabel("Learning Rate")
plt.ylabel("Epochs")

plt.show()