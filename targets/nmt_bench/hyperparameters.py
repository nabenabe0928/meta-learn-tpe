from dataclasses import dataclass


# NOTE: This order must be always same for the query
KEY_ORDER = ["bpe", "n_layers", "n_embed", "n_hidden", "n_heads", "initial_lr"]


@dataclass
class Hyperparameters:
    # Architecture parameters
    bpe: int = 8000
    n_layers: int = 2
    n_embed: int = 512
    n_hidden: int = 1024
    n_heads: int = 8
    initial_lr: float = 0.0006
