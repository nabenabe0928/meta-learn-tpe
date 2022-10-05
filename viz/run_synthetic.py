from argparse import ArgumentParser, Namespace
from typing import Dict, Tuple
import json
import os

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

import numpy as np

from optimizers.tpe.optimizer import RandomOptimizer, TPEOptimizer

from targets.ellipsoid.func import Ellipsoid

from viz.constants import TICK_PARAMS


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["mathtext.fontset"] = "stix"  # The setting of math font
plt.rc("hatch", color="k", linewidth=3)

DIM = 4
MAX_EVALS = 200
N_INDEPENDENT_RUNS = 50
N_METADATA = 100
N_INIT = MAX_EVALS * 5 // 100
N_WARMSTART = N_METADATA * 5 // 100
QUANTILE = 0.1
CENTER_LOCS = [0.0, 1.0, 2.0, 3.0, 4.0]
LOSS_NAME = "loss"
INSET = True
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


def optimize_by_tpe(args: Namespace) -> Tuple[np.ndarray, np.ndarray]:
    opt = TPEOptimizer(seed=args.exp_id, **TPE_PARAMS)
    opt.optimize()
    return opt.fetch_observations()[LOSS_NAME], opt.collect_task_weight_log()


def optimize_by_metalearn_tpe(
    args: Namespace,
    metadata: Dict[str, Dict[str, np.ndarray]],
    warmstart_configs: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    opt = TPEOptimizer(
        seed=args.exp_id,
        uniform_transform=args.uniform_transform,
        dim_reduction_factor=args.dim_reduction_factor,
        metadata=metadata,
        warmstart_configs=warmstart_configs,
        **TPE_PARAMS,
    )
    opt.optimize()
    return opt.fetch_observations()[LOSS_NAME], opt.collect_task_weight_log()


def get_task_key(center: float) -> str:
    return f"center={center:.0f}"


def add_colorbar(axes, cm) -> None:
    zeros = [[MAX_EVALS // 2, 10], [MAX_EVALS, 10]]
    level = np.linspace(0, 1, len(CENTER_LOCS) * 20 + 1)
    cb = axes[0].contourf(zeros, zeros, zeros, level, cmap=cm)
    cbar = fig.colorbar(cb, ax=axes.ravel().tolist(), pad=0.025)
    cbar.ax.set_title("Meta-task", fontsize=16, y=1.01)
    labels = [""] * 100
    labels[-21], labels[18] = "Dissimilar $\\Longleftarrow$", "$\\Longrightarrow$ Similar"
    cbar.set_ticks(np.arange(len(labels)) / (len(labels) - 1))
    cbar.set_ticklabels(labels)
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), rotation=270, va="center")
    cbar.ax.tick_params(size=0)


def plot_curve(
    ax: plt.Axes,
    dx: np.ndarray,
    center: float,
    mean_dict: Dict[str, np.ndarray],
    ste_dict: Dict[str, np.ndarray],
    cm,
    plot_kwargs,
    uniform_kwargs
) -> None:
    key = get_task_key(center)
    m, s = mean_dict[key], ste_dict[key]
    color = cm((center + 0.5) / (len(CENTER_LOCS)))
    ax.plot(dx, m, color=color, lw=2, **plot_kwargs)
    ax.fill_between(dx, m - s, m + s, color=color, alpha=0.1)

    m, s = mean_dict[f"uniform-{key}"], ste_dict[f"uniform-{key}"]
    ax.plot(dx, m, color=color, lw=0.5, **uniform_kwargs)
    ax.fill_between(dx, m - s, m + s, color=color, alpha=0.1)


def get_inset_ax_start(ax: plt.Axes) -> plt.Axes:
    axins = zoomed_inset_axes(
        ax,
        zoom=2.5,
        bbox_to_anchor=(600, 475),
        loc="upper right",
        borderpad=0.1,
        axes_kwargs=dict(aspect=10),
    )

    axins.set_xlim(28, 45)
    axins.set_ylim(10, 100)
    axins.set_yscale("log")
    axins.grid(which="minor", color="gray", linestyle=":")
    axins.grid(which="major", color="black")
    axins.tick_params(**TICK_PARAMS)
    axins.tick_params(axis='y', which='major', labelsize=1)
    axins.tick_params(axis='y', which='minor', labelsize=1)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="gray", ec="red", alpha=0.2)
    return axins


def get_inset_ax_end(ax: plt.Axes) -> plt.Axes:
    axins = zoomed_inset_axes(
        ax,
        zoom=3,
        bbox_to_anchor=(1305, 440),
        loc="upper right",
        borderpad=0.1,
        axes_kwargs=dict(aspect=40),
    )

    axins.set_xlim(150, 200)
    axins.set_ylim(1, 6)
    axins.set_yscale("log")
    axins.grid(which="minor", color="gray", linestyle=":")
    axins.grid(which="major", color="black")
    axins.tick_params(**TICK_PARAMS)
    axins.tick_params(axis='y', which='major', labelsize=1)
    axins.tick_params(axis='y', which='minor', labelsize=1)
    mark_inset(ax, axins, loc1=3, loc2=4, fc="gray", ec="red", alpha=0.2)
    return axins


def plot_task_weight(
    ax: plt.Axes,
    weight_dict: Dict[str, np.ndarray],
):
    cm = plt.get_cmap("gist_rainbow")
    dx = np.arange(MAX_EVALS) + 1
    plot_kwargs = dict(
        marker="*",
        linestyle="dotted",
        markevery=MAX_EVALS // 20,
    )
    uniform_kwargs = dict(
        marker="s",
        linestyle="solid",
        markevery=MAX_EVALS // 20,
        lw=0.5,
    )
    for center in range(5):
        color = cm((center + 0.5) / (len(CENTER_LOCS)))
        ax.plot(dx, 1 - weight_dict[get_task_key(center)], color=color, **plot_kwargs)
        ax.plot(dx, np.full(MAX_EVALS, 0.5), color=color, **uniform_kwargs)

    ax.set_xlabel("Number of config evaluations")
    ax.set_ylabel("$k_t(t_1, t_2)$")
    ax.set_ylim(-0.05, 0.55)
    ax.grid()


def plot_result(
    fig: plt.Figure,
    axes: plt.Axes,
    mean_dict: Dict[str, np.ndarray],
    ste_dict: Dict[str, np.ndarray],
    weight_dict: Dict[str, np.ndarray],
) -> None:
    ax = axes[0]
    dx = np.arange(MAX_EVALS) + 1
    tpe_key = "tpe"
    cm = plt.get_cmap("gist_rainbow")

    m, s = mean_dict[tpe_key], ste_dict[tpe_key]
    mean_dict.pop(tpe_key)
    ste_dict.pop(tpe_key)
    color = "black"
    lines, labels = [], []
    plot_kwargs = dict(linestyle="dashed", marker="", markevery=MAX_EVALS // 20)

    label = "TPE"
    lines.append(ax.plot(dx, m, color=color, label=label, **plot_kwargs)[0])
    labels.append(label)
    ax.fill_between(dx, m - s, m + s, color=color, alpha=0.1)
    if INSET:
        axins1 = get_inset_ax_start(ax)
        axins1.plot(dx, m, color=color, label=label, **plot_kwargs)
        axins1.fill_between(dx, m - s, m + s, color=color, alpha=0.1)
        axins2 = get_inset_ax_end(ax)
        axins2.plot(dx, m, color=color, label=label, **plot_kwargs)
        axins2.fill_between(dx, m - s, m + s, color=color, alpha=0.1)

    plot_kwargs.update(linestyle="dotted", marker="*")
    uniform_kwargs = plot_kwargs.copy()
    uniform_kwargs.update(linestyle="solid", marker="s")
    for center in CENTER_LOCS:
        key = get_task_key(center)
        m, s = mean_dict[key], ste_dict[key]
        plot_curve(ax, dx, center, mean_dict, ste_dict, cm, plot_kwargs=plot_kwargs, uniform_kwargs=uniform_kwargs)
        if INSET:
            plot_curve(
                axins1, dx, center, mean_dict, ste_dict, cm, plot_kwargs=plot_kwargs, uniform_kwargs=uniform_kwargs
            )
            plot_curve(
                axins2, dx, center, mean_dict, ste_dict, cm, plot_kwargs=plot_kwargs, uniform_kwargs=uniform_kwargs
            )

    label = "Meta-learning TPE"
    lines.append(ax.plot(0, 0, color="red", label="Meta-learning TPE", lw=2.0, **plot_kwargs)[0])
    labels.append(label)
    label = "Naïve meta-learning TPE"
    lines.append(ax.plot(0, 0, color="red", label="Naïve meta-learning TPE", lw=0.5, **uniform_kwargs)[0])
    labels.append(label)

    ax.set_yscale("log")
    add_colorbar(axes, cm)
    ax.set_ylabel(r"$f(x)$")
    ax.set_xlim(1, MAX_EVALS)
    ax.grid(which="minor", color="gray", linestyle=":")
    ax.grid(which="major", color="black")
    axes[1].legend(
        handles=lines,
        loc="upper center",
        labels=labels,
        bbox_to_anchor=(0.5, -0.45),
        fancybox=False,
        shadow=False,
        ncol=len(labels),
    )
    plot_task_weight(axes[1], weight_dict)
    plt.savefig("figs/similarity-vs-convergence.pdf", bbox_inches="tight")


def get_mean_and_ste(loss_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # shape: (n_independent_runs, n_evals)
    (n_independent_runs, _) = loss_vals.shape
    cum_loss_vals = np.minimum.accumulate(loss_vals, axis=-1)
    mean = np.mean(cum_loss_vals, axis=0)
    ste = np.std(cum_loss_vals, axis=0) / np.sqrt(n_independent_runs)
    return mean, ste


def collect_data():
    parser = ArgumentParser()
    # parser.add_argument("--exp_id", type=int, required=True)
    # parser.add_argument("--center", type=float, required=True)
    parser.add_argument("--uniform_transform", type=str, choices=["True", "False"], default="False")
    parser.add_argument("--dim_reduction_factor", type=float, default=2.5)

    args = parser.parse_args()
    import time
    start = time.time()

    mean_dict, ste_dict, weight_dict = {}, {}, {}
    loss_vals = np.empty((N_INDEPENDENT_RUNS, MAX_EVALS))
    weights = np.empty((N_INDEPENDENT_RUNS, MAX_EVALS))
    for seed in range(N_INDEPENDENT_RUNS):
        print(f"### Start Optimization {seed + 1}: {time.time() - start:.2f} [sec] ###")
        args.exp_id = seed
        loss_vals[seed], weights[seed] = optimize_by_tpe(args)
    else:
        mean_dict["tpe"], ste_dict["tpe"] = get_mean_and_ste(loss_vals)
        weight_dict["tpe"] = np.median(weights, axis=0)

    loss_vals_for_uniform = np.empty((N_INDEPENDENT_RUNS, MAX_EVALS))
    weights_for_uniform = np.empty((N_INDEPENDENT_RUNS, MAX_EVALS))
    for center in CENTER_LOCS:
        for seed in range(N_INDEPENDENT_RUNS):
            print(f"### Start Optimization {seed + 1}: {time.time() - start:.2f} [sec] ###")
            args.center = center  # TODO: remove
            args.exp_id = seed  # TODO: remove
            metadata = get_metadata(args, Ellipsoid(center=args.center, dim=DIM))
            warmstart_configs = select_warmstart_configs(metadata)

            args.uniform_transform = False
            params = dict(metadata=metadata, warmstart_configs=warmstart_configs)
            loss_vals[seed], weights[seed] = optimize_by_metalearn_tpe(args, **params)
            args.uniform_transform = True
            loss_vals_for_uniform[seed], weights_for_uniform[seed] = optimize_by_metalearn_tpe(args, **params)
        else:
            key = get_task_key(args.center)
            mean_dict[key], ste_dict[key] = get_mean_and_ste(loss_vals)
            weight_dict[key] = np.median(weights, axis=0)
            mean_dict[f"uniform-{key}"], ste_dict[f"uniform-{key}"] = get_mean_and_ste(loss_vals_for_uniform)
            weight_dict[f"uniform-{key}"] = np.median(weights_for_uniform, axis=0)

    data = {
        "mean": {k: v.tolist() for k, v in mean_dict.items()},
        "ste": {k: v.tolist() for k, v in ste_dict.items()},
        "weight": {k: v.tolist() for k, v in weight_dict.items()},
    }
    return data


if __name__ == "__main__":
    FILE_PATH = "results/synthetic.json"
    if os.path.exists(FILE_PATH):
        data = json.load(open(FILE_PATH))
    else:
        data = collect_data()
        with open(FILE_PATH, mode="w") as f:
            json.dump(data, f, indent=4)

    mean_dict = {k: np.array(v) for k, v in data["mean"].items()}
    ste_dict = {k: np.array(v) for k, v in data["ste"].items()}
    weight_dict = {k: np.maximum(np.array(v), 0.5) for k, v in data["weight"].items()}
    fig, axes = plt.subplots(
        nrows=2,
        figsize=(20, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )
    plot_result(fig, axes, mean_dict, ste_dict, weight_dict)
