import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# Load results
file_path = Path("experiments") / "exp2" /"exp2.json"
plots_path = "results/exp2"
os.makedirs(plots_path, exist_ok=True)

with file_path.open() as f:
    results = json.load(f)

df = pd.DataFrame(results)

epochs = [i for i in range(100, 2100, 100)]


for opt in df["optimiser"].unique():
    idx = 0
    opt_df = df[df["optimiser"] == opt]

    for seed in opt_df["seed"]:
        loss_history = list(opt_df["loss_history"][opt_df["seed"] == seed])[0]
        plt.plot(
            epochs,
            loss_history,
            marker="o",
            label=f"loss"
        )

        plt.title(f"Accuracy plot (optimiser: {opt}, seed{seed})")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.savefig(plots_path + f"/{opt}_{idx}_{seed}", )
        plt.close()
        idx+=1

# # Aggregate heatmap (max accuracy across all hidden_sizes)
# agg = df.groupby(["epochs", "lr"])["accuracy"].max().reset_index()

# pivot = agg.pivot(
#     index="epochs",
#     columns="lr",
#     values="accuracy"
# )

# plt.figure(figsize=(10, 6))

# sns.heatmap(
#     pivot,
#     annot=True,
#     fmt=".2f",
#     cmap="plasma",
#     cbar_kws={"label": "Max Accuracy"}
# )

# plt.title("Max Accuracy Heatmap across Hidden Sizes")
# plt.xlabel("Learning Rate")
# plt.ylabel("Epochs")

# # 3D grid search visualization
# fig = px.scatter_3d(
#     df,
#     x="hidden_size",
#     y="lr",
#     z="epochs",
#     color="accuracy",
#     color_continuous_scale="inferno",
#     hover_data=["hidden_size", "lr", "epochs", "accuracy"]
# )

# fig.update_layout(
#     title="Grid Search: Accuracy across Hidden Size, LR, Epochs"
# # )

# fig.show()

# # Heatmaps per hidden_size
# for hs in sorted(df["hidden_size"].unique()):

#     subset = df[df["hidden_size"] == hs].pivot_table(
#         index="epochs",
#         columns="lr",
#         values="accuracy",
#         aggfunc="max"
#     )

#     plt.figure(figsize=(10, 6))

#     sns.heatmap(
#         subset,
#         annot=True,
#         fmt=".2f",
#         cmap="plasma",
#         cbar_kws={"label": "Accuracy"}
#     )

#     plt.title(f"Accuracy Heatmap (hidden_size={hs})")
#     plt.xlabel("Learning Rate")
#     plt.ylabel("Epochs")

#     plt.show()
# plt.show()


# # Accuracy vs Epochs (per learning rate)

# for lr in sorted(df["lr"].unique()):

#     plt.figure(figsize=(10, 6))

#     subset = df[df["lr"] == lr]

#     for hs in sorted(subset["hidden_size"].unique()):

#         hs_subset = subset[subset["hidden_size"] == hs]
#         hs_subset = hs_subset.sort_values(by="epochs")
