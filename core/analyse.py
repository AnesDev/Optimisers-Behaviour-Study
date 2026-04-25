import json
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# -------- LOAD --------
file_path = Path("experiments") / "exp2" / "exp2.json"
plots_path = Path("results") / "exp2"
plots_path.mkdir(parents=True, exist_ok=True)

with file_path.open() as f:
    results = json.load(f)

df = pd.DataFrame(results)

# -------- COMPUTE DELTAS --------
delta_rows = []

grouped = df.groupby(["optimizer", "seed"])

for (opt, seed), group in grouped:

    base_rows = group[group["transform"] == "none"]

    if len(base_rows) == 0:
        continue  # safety

    base_acc = base_rows.iloc[0]["accuracy"]

    for _, row in group.iterrows():
        delta_rows.append({
            "optimizer": opt,
            "seed": seed,
            "transform": row["transform"],
            "delta_acc": row["accuracy"] - base_acc
        })

df_delta = pd.DataFrame(delta_rows)

# -------- PLOT --------
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df_delta,
    x="transform",
    y="delta_acc",
    hue="optimizer"
)

plt.title("Sensitivity to Parameterization (Δ Accuracy vs baseline)")
plt.ylabel("Accuracy difference")
plt.xlabel("Transformation")

plt.tight_layout()
plt.savefig(plots_path / "delta_accuracy_boxplot.png")
plt.show()