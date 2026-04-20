from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np

import sys
from pathlib import Path 
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.mlp import MLP
from optimisers.adam import Adam

import itertools
from multiprocessing import Pool
import json
import time

from pathlib import Path


X, y = make_moons(n_samples=500, noise=0.2, random_state=42)

encoder = OneHotEncoder(sparse_output=False)
Y = encoder.fit_transform(y.reshape(-1,1))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


def train_and_evaluate(params):

    hidden_size, lr, epochs = params
    optimiser = Adam()
    mlp = MLP(input_size=2, hidden_size=hidden_size, output_size=2, optimizer=optimiser)
    mlp.train(X_train, Y_train, epochs=epochs)
    predictions = mlp.predict(X_test)

    accuracy = np.mean(predictions == np.argmax(Y_test, axis=1))
    return {
        "hidden_size": int(hidden_size),
        "lr": float(lr),
        "epochs": int(epochs),
        "accuracy": float(accuracy)
    }

if __name__ == "__main__":

    start = time.perf_counter()

    param_grid = {
        "hidden_size": [5 + i for i in np.arange(0, 55, 5)],
        "lr": [0.001 + i for i in np.arange(0, 0.011, 0.001)],
        "epochs":[500 + j for j in np.arange(0, 3500, 500)]
    }

    combinations = list(itertools.product(
        param_grid["hidden_size"],
        param_grid["lr"],
        param_grid["epochs"]
    ))

    with Pool() as pool:
        results = pool.map(train_and_evaluate, combinations)

    end = time.perf_counter()

    print(f"Total time: {end - start:.2f} seconds")
    print(f"Avg time per model: {(end - start)/len(combinations):.4f} seconds")

    output_path = Path(__file__).parent.parent / "experiments" / "exp1" / "exp1.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)