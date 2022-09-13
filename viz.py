from typing import Any, List, Tuple
import json

from eaf import (
    get_empirical_attainment_surface,
    EmpiricalAttainmentFuncPlot,
)

import matplotlib.pyplot as plt

import numpy as np


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.rcParams["mathtext.fontset"] = "stix"  # The setting of math font

N_SAMPLES = 100
N_INIT = N_SAMPLES * 5 // 100
N_RUNS = 20
N_OBJ = 2
COSTS_SHAPE = (N_RUNS, N_SAMPLES, N_OBJ)
# LEVELS = [N_RUNS // 4, N_RUNS // 2, (3 * N_RUNS) // 4]
LEVELS = [N_RUNS // 2] * 3
WARMSTART_OPT = "tpe_q=0.10_df=3"
COLOR_LABEL_DICT = {
    "random": ("olive", "Random"),
    "naive_metalearn_tpe": ("green", "Uniform weight"),
    "normal_tpe": ("blue", "MOTPE BW0.5"),
    "tpe_q=0.10_df=3": ("red", "Meta-learn TPE df=4"),
    "warmstart_config": ("black", "Warmstart configs"),
}
BENCH_NAMES = ["hpolib", "nmt"]
DATASET_NAMES = {
    "hpolib": [
        "naval_propulsion",
        "parkinsons_telemonitoring",
        "protein_structure",
        "slice_localization",
    ],
    "nmt": [
        "so_en",
        "sw_en",
        "tl_en",
    ],
}
OBJ_NAMES_DICT = {
    "hpolib": ["runtime", "valid_mse"],
    "nmt": ["decoding_time", "bleu"],
}
LARGER_IS_BETTER_DICT = {
    "hpolib": None,
    "nmt": [1],
}
LOGSCALE_DICT = {
    "hpolib": [1],
    "nmt": None,
}


def get_costs(obj_names: List[str], dataset_name: str, opt_name: str) -> None:
    n_samples = N_SAMPLES if opt_name != "warmstart_config" else N_INIT
    costs = np.empty((N_RUNS, n_samples, N_OBJ))
    opt_name = opt_name if opt_name != "warmstart_config" else WARMSTART_OPT
    for i in range(N_RUNS):
        data = json.load(open(f"results/{dataset_name}/{opt_name}/{i:0>2}.json"))
        costs[i, :, 0] = data[obj_names[0]][:n_samples]
        costs[i, :, 1] = data[obj_names[1]][:n_samples]

    return costs


def get_true_pareto_front_and_ref_point(
    obj_names: List[str],
    bench_name: str,
    dataset_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    _bench_name = "nmt_bench" if bench_name == "nmt" else bench_name
    data = json.load(open(f"targets/{_bench_name}/pareto-fronts/{dataset_name}.json"))
    ref_point = np.asarray([
        json.load(open(f"targets/{_bench_name}/reference-point/{dataset_name}.json"))[name]
        for name in obj_names
    ])
    true_pf = np.empty((len(data[obj_names[0]]), N_OBJ))
    true_pf[:, 0], true_pf[:, 1] = data[obj_names[0]], data[obj_names[1]]
    return true_pf, ref_point


def plot_eaf(
    ax: plt.Axes,
    eaf_plot: EmpiricalAttainmentFuncPlot,
    obj_names: List[str],
    dataset_name: str,
    **kwargs
) -> Tuple[List[Any], List[str]]:
    surfs_list, colors, labels = [], [], []
    for opt_name, color_label in COLOR_LABEL_DICT.items():
        color, label = color_label
        colors.append(color)
        labels.append(label)
        costs = get_costs(obj_names, dataset_name, opt_name)
        surfs = get_empirical_attainment_surface(costs.copy(), levels=LEVELS, **kwargs)
        surfs_list.append(surfs)
    else:
        lines = eaf_plot.plot_multiple_surface_with_band(ax, surfs_list=surfs_list, colors=colors, labels=labels)
        label = "True Pareto front"
        lines.append(eaf_plot.plot_true_pareto_surface(ax, color="black", label=label, linestyle="--", marker="*"))
        labels.append(label)

    ax.set_xlabel(obj_names[0])
    ax.set_ylabel(obj_names[1])
    return lines, labels


def plot_hv(
    ax: plt.Axes,
    eaf_plot: EmpiricalAttainmentFuncPlot,
    obj_names: List[str],
    dataset_name: str,
    log: bool,
    **kwargs
) -> Tuple[List[Any], List[str]]:

    color_label_dict = {k: v for k, v in COLOR_LABEL_DICT.items() if k != "warmstart_config"}
    n_opts = len(color_label_dict)
    costs_array, colors, labels = np.empty((n_opts, *COSTS_SHAPE)), [], []
    for idx, (opt_name, color_label) in enumerate(color_label_dict.items()):
        color, label = color_label
        colors.append(color)
        labels.append(label)
        costs_array[idx] = get_costs(obj_names, dataset_name, opt_name)
    else:
        label = "True Pareto front"
        lines = eaf_plot.plot_multiple_hypervolume2d_with_band(ax, costs_array, colors, labels, log=log)
        lines.append(
            eaf_plot.plot_true_pareto_surface_hypervolume2d(
                ax, n_observations=N_SAMPLES, color="black", label=label, linestyle="--"
            )
        )
        labels.append(label)

        from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

        y_min = np.min([np.min(line._y[90:]) for line in lines if hasattr(line, "_y")])
        y_max = np.max([np.max(line._y[90:]) for line in lines if hasattr(line, "_y")])
        axins = zoomed_inset_axes(ax, zoom=2, loc="upper right")
        eaf_plot.plot_multiple_hypervolume2d_with_band(axins, costs_array, colors, labels, log=log)
        axins.set_xlim(90, 100)
        axins.set_ylim(y_min, y_max)
        return lines, labels


def plot(
    ax: plt.Axes,
    bench_id: int,
    data_id: int,
    hv_mode: bool,
) -> List[Any]:
    bench_name = BENCH_NAMES[bench_id]
    dataset_name = DATASET_NAMES[bench_name][data_id]
    obj_names = OBJ_NAMES_DICT[bench_name]
    kwargs = dict(
        larger_is_better_objectives=LARGER_IS_BETTER_DICT[bench_name],
        log_scale=LOGSCALE_DICT[bench_name]
    )

    true_pf, ref_point = get_true_pareto_front_and_ref_point(obj_names, bench_name, dataset_name)
    eaf_plot = EmpiricalAttainmentFuncPlot(true_pareto_sols=true_pf, ref_point=ref_point, **kwargs)
    if hv_mode:
        lines, labels = plot_hv(ax, eaf_plot, obj_names, dataset_name, log=False, **kwargs)
    else:
        lines, labels = plot_eaf(ax, eaf_plot, obj_names, dataset_name, **kwargs)

    ax.set_title(dataset_name)
    ax.grid(which="minor", color="gray", linestyle=":")
    ax.grid(which="major", color="black")

    return lines, labels


if __name__ == "__main__":
    hv_mode = False
    for bench_id, n in enumerate([4, 3]):
        _, axes = plt.subplots(nrows=2, ncols=2)
        for data_id in range(n):
            r, c = data_id // 2, data_id % 2
            lines, labels = plot(axes[r][c], bench_id=bench_id, data_id=data_id, hv_mode=hv_mode)
        else:
            axes[-1][0].legend(
                handles=lines,
                loc='upper center',
                labels=labels,
                fontsize=18,
                bbox_to_anchor=(1, -0.15),
                fancybox=False,
                shadow=False,
                ncol=len(labels)
            )

        plt.show()
