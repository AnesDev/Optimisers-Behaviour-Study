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
