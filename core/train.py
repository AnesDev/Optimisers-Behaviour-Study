from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import numpy as np

from models.mlp import MLP
from models.swap_mlp import swap_MLP
from optimisers.adam import Adam
from .config import OPTIMIZERS, SEEDS
from .transformations import TRANSFORMS

import itertools
from tqdm import tqdm
from multiprocessing import Pool
import json
import time
from pathlib import Path

X, y = make_moons(n_samples=500, noise=0.2, random_state=42)

encoder = OneHotEncoder(sparse_output=False)
Y = encoder.fit_transform(y.reshape(-1,1))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

def init_params(input_size, hidden_size, output_size, seed):
    np.random.seed(seed)
    return {
        "w1": np.random.randn(input_size, hidden_size),
        "b1": np.random.randn(1, hidden_size),
        "w2": np.random.randn(hidden_size, output_size),
        "b2": np.random.randn(1, output_size),
    }

def train_and_evaluate(args):
    opt_name, opt_class, t_name, transform, seed = args

    np.random.seed(seed)

    optimizer = opt_class(lr=0.001)

    # ---- INIT PARAMS DIRECTLY ----
    params = init_params(2, 20, 2, seed)

    # ---- APPLY TRANSFORM ----
    params = transform(params, seed)

    # ---- BUILD MODEL ----
    mlp = MLP(
        input_size=2,
        hidden_size=20,
        output_size=2,
        optimizer=optimizer,
        params=params
    )

    loss_history = mlp.train(X_train, Y_train, epochs=2000)

    preds = mlp.predict(X_test)
    acc = np.mean(preds == np.argmax(Y_test, axis=1))

    return {
        "optimizer": opt_name,
        "transform": t_name,
        "seed": seed,
        "accuracy": float(acc),
        "loss_history": loss_history
    }

if __name__ == "__main__":

    start = time.perf_counter()

    runs = [
        (opt_name, opt_class, t_name, transform, seed)
        for opt_name, opt_class in OPTIMIZERS
        for t_name, transform in TRANSFORMS
        for seed in SEEDS
    ]


    with Pool() as pool:
        results = list(tqdm(pool.imap(train_and_evaluate, runs), total=len(runs)))
    end = time.perf_counter()

    print(f"Total time: {end - start:.2f} seconds")
    print(f"Avg time per model: {(end - start)/len(runs):.4f} seconds")

    output_path = Path(__file__).parent.parent / "experiments" / "exp2" / "exp2.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)