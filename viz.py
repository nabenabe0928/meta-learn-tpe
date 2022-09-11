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

_, ax = plt.subplots()
color_label_dict = {
    "random": ("olive", "Random"),
    "normal_tpe": ("black", "TPE"),
    "naive_metalearn_tpe": ("blue", "Uniform weight"),
    "tpe_q=0.10_df=5": ("red", "Meta-learn TPE"),
}
bench_id, data_id = 0, 2

bench_name = ["hpolib", "nmt"][bench_id]
dataset_name = {
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
}[bench_name][data_id]
obj_names = {
    "hpolib": ["runtime", "valid_mse"],
    "nmt": ["decoding_time", "bleu"],
}[bench_name]
larger_is_better = {
    "hpolib": None,
    "nmt": [1],
}[bench_name]
kwargs = dict(
    larger_is_better_objectives=larger_is_better,
    log_scale=[1] if bench_name == "hpolib" else None
)

_bench_name = "nmt_bench" if bench_name == "nmt" else bench_name
data = json.load(open(f"targets/{_bench_name}/pareto-fronts/{dataset_name}.json"))
true_pf = np.empty((len(data[obj_names[0]]), 2))
true_pf[:, 0] = data[obj_names[0]]
true_pf[:, 1] = data[obj_names[1]]
eaf_plot = EmpiricalAttainmentFuncPlot(
    true_pareto_sols=true_pf,
    **kwargs,
)

surfs_list, colors, labels = [], [], []
for opt_name, color_label in color_label_dict.items():
    color, label = color_label
    colors.append(color)
    labels.append(label)
    costs = np.empty((20, 100, 2))
    for i in range(20):
        data = json.load(open(f"results/{dataset_name}/{opt_name}/{i:0>2}.json"))
        costs[i, :, 0] = data[obj_names[0]]
        costs[i, :, 1] = data[obj_names[1]]

    surfs = get_empirical_attainment_surface(costs, levels=[5, 10, 15], **kwargs)
    surfs_list.append(surfs)
else:
    eaf_plot.plot_multiple_surface_with_band(ax, surfs_list=surfs_list, colors=colors, labels=labels)
    eaf_plot.plot_true_pareto_surface(ax, color="black", label="True Pareto front", linestyle="--")

ax.set_xlabel(obj_names[0])
ax.set_ylabel(obj_names[1])
ax.grid(which="minor", color="gray", linestyle=":")
ax.grid(which="major", color="black")
ax.legend()
plt.show()
