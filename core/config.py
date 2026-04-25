from optimisers.adam import Adam
from optimisers.momentum import Momentum
from optimisers.rmsprop import RMSProp
from optimisers.gd import SGD

OPTIMIZERS = [
    ("sgd", SGD),
    ("momentum", Momentum),
    ("rmsprop", RMSProp),
    ("adam", Adam),
]

SEEDS = list(range(10))

MODEL_CONFIG = {
    "input_size": 2,
    "hidden_size": 20,
    "output_size": 2
}