from dataclasses import dataclass


# NOTE: This order must be always same for the query
KEY_ORDER = ["alpha", "batch_size", "depth", "learning_rate_init", "width", "seed"]


@dataclass
class Hyperparameters:
    # Architecture parameters
    depth: int = 2
    width: int = 101
    # Training parameters
    alpha: float = 3.5938137e-5
    batch_size: int = 25
    learning_rate_init: float = 1.6681006e-3
