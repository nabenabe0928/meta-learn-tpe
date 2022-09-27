from argparse import ArgumentParser, Namespace
from typing import Dict, Tuple

import matplotlib.pyplot as plt

import numpy as np

from optimizers.tpe.optimizer import RandomOptimizer, TPEOptimizer

from targets.ellipsoid.func import Ellipsoid


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["mathtext.fontset"] = "stix"  # The setting of math font

DIM = 4
MAX_EVALS = 200
N_INDEPENDENT_RUNS = 20
N_METADATA = 100
N_INIT = MAX_EVALS * 5 // 100
N_WARMSTART = N_METADATA * 5 // 100
QUANTILE = 0.1
CENTER_LOCS = [0.0, 1.0, 2.0, 3.0, 4.0]
LOSS_NAME = "loss"
TARGET_BENCH = Ellipsoid(center=0.0, dim=DIM)

TPE_PARAMS = dict(
    obj_func=TARGET_BENCH.objective_func,
    config_space=TARGET_BENCH.config_space,
    max_evals=MAX_EVALS,
    n_init=N_INIT,
    quantile=QUANTILE,
)


def get_metadata(args: Namespace, bench: Ellipsoid) -> Dict[str, Dict[str, np.ndarray]]:
    opt = RandomOptimizer(
        obj_func=bench.objective_func,
        config_space=bench.config_space,
        max_evals=N_METADATA,
        seed=args.exp_id,
    )
    opt.optimize()
    data = opt.fetch_observations()
    data.pop(opt._runtime_name)
    return {f"center={bench.center}": data}


def select_warmstart_configs(metadata: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    warmstart_configs: Dict[str, np.ndarray] = {}
    for data in metadata.values():
        order = np.argsort(data[LOSS_NAME])[:N_WARMSTART]
        warmstart_configs = {k: v[order] for k, v in data.items()}

    return warmstart_configs


def optimize_by_tpe(args: Namespace) -> np.ndarray:
    opt = TPEOptimizer(seed=args.exp_id, **TPE_PARAMS)
    opt.optimize()
    return opt.fetch_observations()[LOSS_NAME]


def optimize_by_metalearn_tpe(
    args: Namespace,
    metadata: Dict[str, Dict[str, np.ndarray]],
    warmstart_configs: Dict[str, np.ndarray],
) -> np.ndarray:
    opt = TPEOptimizer(
        seed=args.exp_id,
        # uniform_transform=args.uniform_transform,
        dim_reduction_factor=args.dim_reduction_factor,
        metadata=metadata,
        warmstart_configs=warmstart_configs,
        **TPE_PARAMS,
    )
    opt.optimize()
    return opt.fetch_observations()[LOSS_NAME]


def get_task_key(center: float) -> str:
    return f"center={center:.0f}"


def add_colorbar(ax, cm) -> None:
    zeros = [[0, 0], [0, 0]]
    level = np.linspace(0, 1, len(CENTER_LOCS) * 20 + 1)
    cb = ax.contourf(zeros, zeros, zeros, level, cmap=cm)
    cbar = fig.colorbar(cb)
    cbar.ax.set_title("Meta-task", fontsize=14, y=1.01)
    labels = [""] * 100
    labels[-21], labels[18] = "Dissimilar $\\Longleftarrow$", "$\\Longrightarrow$ Similar"
    cbar.set_ticks(np.arange(len(labels)) / (len(labels) - 1))
    cbar.set_ticklabels(labels)
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), rotation=270, va="center")
    cbar.ax.tick_params(size=0)


def plot_result(
    fig: plt.Figure,
    ax: plt.Axes,
    mean_dict: Dict[str, np.ndarray],
    ste_dict: Dict[str, np.ndarray],
) -> None:
    dx = np.arange(MAX_EVALS) + 1
    tpe_key = "tpe"
    cm = plt.get_cmap("gist_rainbow")

    m, s = mean_dict[tpe_key], ste_dict[tpe_key]
    mean_dict.pop(tpe_key)
    ste_dict.pop(tpe_key)
    color = "black"
    plot_kwargs = dict(linestyle="dotted", marker="*", markevery=MAX_EVALS // 20)
    ax.plot(dx, m, color=color, label="TPE", **plot_kwargs)
    ax.fill_between(dx, m - s, m + s, color=color, alpha=0.2)

    plot_kwargs.update(linestyle="dashed")
    for center in CENTER_LOCS:
        key = get_task_key(center)
        m, s = mean_dict[key], ste_dict[key]
        color = cm((center + 0.5) / (len(CENTER_LOCS)))
        ax.plot(dx, m, color=color, **plot_kwargs)
        ax.fill_between(dx, m - s, m + s, color=color, alpha=0.2)

    ax.set_yscale("log")
    add_colorbar(ax, cm)
    ax.set_xlabel("Number of config evaluations")
    ax.set_ylabel("$f(x)$")
    ax.set_xlim(1, MAX_EVALS)
    ax.grid(which="minor", color="gray", linestyle=":")
    ax.grid(which="major", color="black")
    ax.legend(loc="upper right")
    plt.savefig("figs/similarity-vs-convergence.pdf", bbox_inches="tight")


def get_mean_and_ste(loss_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # shape: (n_independent_runs, n_evals)
    (n_independent_runs, _) = loss_vals.shape
    cum_loss_vals = np.minimum.accumulate(loss_vals, axis=-1)
    mean = np.mean(cum_loss_vals, axis=0)
    ste = np.std(cum_loss_vals, axis=0) / np.sqrt(n_independent_runs)
    return mean, ste


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--exp_id", type=int, required=True)
    # parser.add_argument("--center", type=float, required=True)
    # parser.add_argument("--uniform_transform", type=str, choices=["True", "False"], default="False")
    parser.add_argument("--dim_reduction_factor", type=float, default=2.5)

    args = parser.parse_args()
    import time
    start = time.time()

    mean_dict, ste_dict = {}, {}
    loss_vals = np.empty((N_INDEPENDENT_RUNS, MAX_EVALS))
    for seed in range(N_INDEPENDENT_RUNS):
        print(f"### Start Optimization {seed + 1}: {time.time() - start:.2f} [sec] ###")
        args.exp_id = seed
        loss_vals[seed] = optimize_by_tpe(args)
    else:
        mean_dict["tpe"], ste_dict["tpe"] = get_mean_and_ste(loss_vals)

    for center in CENTER_LOCS:
        for seed in range(N_INDEPENDENT_RUNS):
            print(f"### Start Optimization {seed + 1}: {time.time() - start:.2f} [sec] ###")
            args.center = center  # TODO: remove
            args.exp_id = seed  # TODO: remove
            metadata = get_metadata(args, Ellipsoid(center=args.center, dim=DIM))
            warmstart_configs = select_warmstart_configs(metadata)

            loss_vals[seed] = optimize_by_metalearn_tpe(
                args, metadata=metadata, warmstart_configs=warmstart_configs
            )
        else:
            key = get_task_key(args.center)
            mean_dict[key], ste_dict[key] = get_mean_and_ste(loss_vals)

    fig, ax = plt.subplots(figsize=(15, 5))
    plot_result(fig, ax, mean_dict, ste_dict)
