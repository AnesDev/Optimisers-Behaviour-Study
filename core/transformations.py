import numpy as np

def permute_hidden_layer(params, seed):
    np.random.seed(seed)

    perm = np.random.permutation(params["w1"].shape[1])

    params["w1"] = params["w1"][:, perm]
    params["b1"] = params["b1"][:, perm]
    params["w2"] = params["w2"][perm, :]

    return params


def scale_hidden_layer(params, seed):
    np.random.seed(seed)

    hidden = params["w1"].shape[1]
    scales = np.random.uniform(0.5, 2.0, size=(hidden,))

    params["w1"] *= scales
    params["b1"] *= scales
    params["w2"] /= scales.reshape(-1, 1)

    return params

def identity(params, seed):
    return {k: v.copy() for k, v in params.items()}


TRANSFORMS = [
    ("none", identity),
    ("permute", permute_hidden_layer),
    ("scale", scale_hidden_layer)
]