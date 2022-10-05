from typing import Dict
import json
import pickle

from fast_pareto import nondominated_rank

import matplotlib.pyplot as plt

import numpy as np

from targets.hpolib.api import DatasetChoices as HPOlibChoices
from targets.nmt_bench.api import DatasetChoices as NMTChoices


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["mathtext.fontset"] = "stix"  # The setting of math font


def get_nd_rank_for_hpolib(percentile: int) -> Dict[str, np.ndarray]:
    nd_rank = {}
    for dataset in HPOlibChoices:
        print(dataset.name)
        costs = pickle.load(open(f"targets/hpolib/metric_vals/{dataset.name}.pkl", "rb"))
        data = np.asarray([costs["valid_mse"], costs["runtime"]]).T
        nd_rank[dataset.name] = nondominated_rank(costs=data)

    return nd_rank


def get_nd_rank_for_nmt(percentile: int) -> Dict[str, np.ndarray]:
    nd_rank = {}
    for dataset in NMTChoices:
        costs = json.load(open(f"nmt-bench/{dataset.value}"))
        data = np.asarray([costs["bleu"], costs["decoding_time"]]).T
        nd_rank[dataset.name] = nondominated_rank(costs=data, larger_is_better_objectives=[0])

    return nd_rank


def plot_cum(ax: plt.Axes, nd_rank: Dict[str, np.ndarray], percentile: int, set_ylabel: bool) -> None:
    colors = ["red", "blue", "green", "purple"]
    for i, (k, v) in enumerate(nd_rank.items()):
        n_configs = v.size
        order = np.argsort(v)[:int(n_configs * percentile / 100)]
        cnt = np.zeros(n_configs)
        cnt[np.arange(n_configs)[order]] = 1
        if len(nd_rank) == 4:
            dataset_name = " ".join([s.capitalize() for s in k.split("_")])
        else:
            lang = {"so": "Somali", "sw": "Swahili", "tl": "Tagalog", "en": "English"}
            dataset_name = " to ".join([lang[s] for s in k.split("_")])
        ax.plot(np.arange(n_configs), np.cumsum(cnt), label=dataset_name, color=colors[i])

    title = f"Cumulated count of Top-{percentile}% configuration"
    ax.set_title(title)
    ax.set_xlabel("Config indices")

    if set_ylabel:
        ax.set_ylabel("Cumulated count")

    ax.legend()
    ax.grid()


if __name__ == "__main__":
    _, axes = plt.subplots(
        figsize=(20, 5),
        ncols=2,
        gridspec_kw={"wspace": 0.1},
    )
    nd_rank = get_nd_rank_for_hpolib(percentile=1)
    plot_cum(axes[0], nd_rank, percentile=1, set_ylabel=True)

    nd_rank = get_nd_rank_for_nmt(percentile=5)
    plot_cum(axes[1], nd_rank, percentile=5, set_ylabel=False)

    plt.savefig("figs/dataset-dist.pdf", bbox_inches="tight")
