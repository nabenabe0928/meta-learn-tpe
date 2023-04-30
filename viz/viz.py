import os
from typing import Any, List, Tuple

from eaf import (
    get_empirical_attainment_surface,
    EmpiricalAttainmentFuncPlot,
)

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

import numpy as np

from viz.constants import (
    BENCH_NAMES,
    COLOR_LABEL_DICT,
    COSTS_SHAPE,
    DATASET_NAMES,
    EAF_ANCHOR_DICT,
    EAF_ASPECT_DICT,
    EAF_INSET_XLIM_DICT,
    EAF_INSET_YLIM_DICT,
    EAF_ZOOM_DICT,
    LARGER_IS_BETTER_DICT,
    LEVELS,
    LINESTYLES_DICT,
    LOGSCALE_DICT,
    MARKER_DICT,
    N_SAMPLES,
    NAME_DICT,
    OBJ_LABEL_DICT,
    OBJ_NAMES_DICT,
    TICK_PARAMS,
)
from viz.utils import get_costs, get_true_pareto_front_and_ref_point


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["mathtext.fontset"] = "stix"  # The setting of math font
PLOT_CHECK_MODE = False
HV_MODE = False


def disable_axis_label(
    ax: plt.Axes,
    set_xlabel: bool = False,
    set_ylabel: bool = False,
) -> None:
    if not set_xlabel:
        ax.set_xlabel("")
    if not set_ylabel:
        ax.set_ylabel("")


def add_inset_axis_for_eaf(
    ax: plt.Axes,
    dataset_name: str,
    eaf_plot: EmpiricalAttainmentFuncPlot,
    surfs_list: List[np.ndarray],
    **plot_kwargs,
) -> None:

    axins = zoomed_inset_axes(
        ax,
        zoom=EAF_ZOOM_DICT[dataset_name],
        bbox_to_anchor=EAF_ANCHOR_DICT[dataset_name],
        bbox_transform=ax.transAxes,
        loc="upper right",
        borderpad=0.1,
        axes_kwargs=dict(aspect=EAF_ASPECT_DICT[dataset_name]),
    )

    eaf_plot.plot_multiple_surface_with_band(axins, surfs_list=surfs_list, **plot_kwargs)
    label = "True Pareto front"
    eaf_plot.plot_true_pareto_surface(axins, color="black", label=label, linestyle="--", marker="*", alpha=0.2)

    axins.set_xlim(EAF_INSET_XLIM_DICT[dataset_name])
    axins.set_ylim(EAF_INSET_YLIM_DICT[dataset_name])
    axins.grid(which="minor", color="gray", linestyle=":")
    axins.grid(which="major", color="black")
    axins.tick_params(**TICK_PARAMS)
    axins.tick_params(axis='y', which='major', labelsize=1)
    axins.tick_params(axis='y', which='minor', labelsize=1)
    disable_axis_label(axins)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="gray", ec="red", alpha=0.2)


def plot_eaf(
    ax: plt.Axes,
    eaf_plot: EmpiricalAttainmentFuncPlot,
    obj_names: List[str],
    dataset_name: str,
    set_xlabel: bool,
    set_ylabel: bool,
    **kwargs
) -> Tuple[List[Any], List[str]]:
    surfs_list, colors, labels, markers, linestyles = [], [], [], [], []
    for opt_name, color_label in COLOR_LABEL_DICT.items():
        color, label = color_label
        colors.append(color)
        labels.append(label)
        linestyles.append(LINESTYLES_DICT[opt_name])
        markers.append(MARKER_DICT[opt_name])
        costs = get_costs(obj_names, dataset_name, opt_name)
        surfs = get_empirical_attainment_surface(costs.copy(), levels=LEVELS, **kwargs)
        surfs_list.append(surfs)
    else:
        plot_kwargs = dict(
            colors=colors, labels=labels, linestyles=linestyles, markers=markers, markersize=3, alpha=0.5
        )
        lines = eaf_plot.plot_multiple_surface_with_band(ax, surfs_list=surfs_list, **plot_kwargs)
        label = "True Pareto front"
        lines.append(eaf_plot.plot_true_pareto_surface(
            ax, color="black", label=label, linestyle="--", marker="*", alpha=0.2
        ))
        labels.append(label)

        if not dataset_name.endswith("_en"):
            add_inset_axis_for_eaf(ax, dataset_name, eaf_plot, surfs_list, **plot_kwargs)

    if set_xlabel:
        ax.set_xlabel(OBJ_LABEL_DICT[obj_names[0]])
    if set_ylabel:
        ax.set_ylabel(OBJ_LABEL_DICT[obj_names[1]])
    return lines, labels


def add_inset_axis_for_hv(
    ax: plt.Axes,
    lines: List[Any],
    eaf_plot: EmpiricalAttainmentFuncPlot,
    costs_array: np.ndarray,
    log: bool,
    **plot_kwargs,
) -> None:

    x_min = 20
    y_max = max(line._y[-1] for line in lines if hasattr(line, "_y"))
    ylim = ax.get_ylim()
    axins = zoomed_inset_axes(
        ax,
        zoom=2.5,
        bbox_to_anchor=(1.76, 0.01),
        bbox_transform=ax.transAxes,
        loc="lower right",
        borderpad=0.1,
        axes_kwargs=dict(aspect=200/(ylim[1] - ylim[0])),
    )
    eaf_plot.plot_multiple_hypervolume2d_with_band(axins, costs_array, log=log, **plot_kwargs)
    axins.set_xlim(x_min, 100)
    axins.set_ylim(y_max - 0.04, y_max + 0.005)
    axins.tick_params(**TICK_PARAMS)
    disable_axis_label(axins)
    mark_inset(ax, axins, loc1=3, loc2=1, fc="gray", ec="red", alpha=0.2)


def plot_hv(
    ax: plt.Axes,
    eaf_plot: EmpiricalAttainmentFuncPlot,
    obj_names: List[str],
    dataset_name: str,
    log: bool,
    set_xlabel: bool,
    set_ylabel: bool,
    **kwargs
) -> Tuple[List[Any], List[str]]:

    n_opts = len(COLOR_LABEL_DICT)
    costs_array, colors, labels, markers, linestyles = np.empty((n_opts, *COSTS_SHAPE)), [], [], [], []
    for idx, (opt_name, color_label) in enumerate(COLOR_LABEL_DICT.items()):
        color, label = color_label
        colors.append(color)
        labels.append(label)
        markers.append(MARKER_DICT[opt_name])
        linestyles.append(LINESTYLES_DICT[opt_name])
        costs_array[idx] = get_costs(obj_names, dataset_name, opt_name)
    else:
        label = "True Pareto front"
        plot_kwargs = dict(colors=colors, labels=labels, markers=markers, markevery=5, linestyles=linestyles)
        lines = eaf_plot.plot_multiple_hypervolume2d_with_band(ax, costs_array, log=log, **plot_kwargs)
        lines.append(
            eaf_plot.plot_true_pareto_surface_hypervolume2d(
                ax, n_observations=N_SAMPLES, color="black", label=label, linestyle="--"
            )
        )
        labels.append(label)
        ax.set_ylim(ymin=0.78, ymax=1.01)

        if not dataset_name.endswith("en"):
            add_inset_axis_for_hv(ax, lines, eaf_plot, costs_array, log, **plot_kwargs)

        disable_axis_label(ax, set_xlabel=set_xlabel, set_ylabel=set_ylabel)
        return lines, labels


def plot(
    ax: plt.Axes,
    bench_id: int,
    data_id: int,
    hv_mode: bool,
    set_xlabel: bool,
    set_ylabel: bool,
) -> List[Any]:
    bench_name = BENCH_NAMES[bench_id]
    dataset_name = DATASET_NAMES[bench_name][data_id]
    obj_names = OBJ_NAMES_DICT[bench_name]
    kwargs = dict(
        larger_is_better_objectives=LARGER_IS_BETTER_DICT[bench_name],
        log_scale=LOGSCALE_DICT[bench_name],
    )

    true_pf, ref_point = get_true_pareto_front_and_ref_point(obj_names, bench_name, dataset_name)
    eaf_plot = EmpiricalAttainmentFuncPlot(true_pareto_sols=true_pf, ref_point=ref_point, **kwargs)
    kwargs.update(set_xlabel=set_xlabel, set_ylabel=set_ylabel)
    if hv_mode:
        lines, labels = plot_hv(ax, eaf_plot, obj_names, dataset_name, log=False, **kwargs)
    else:
        lines, labels = plot_eaf(ax, eaf_plot, obj_names, dataset_name, **kwargs)

    ax.set_title(NAME_DICT[dataset_name])
    ax.grid(which="minor", color="gray", linestyle=":")
    ax.grid(which="major", color="black")

    return lines, labels


def plot_hv_for_hpolib(subplots_kwargs, legend_kwargs, hv_mode: bool) -> None:
    _, axes = plt.subplots(**subplots_kwargs)
    for data_id in range(4):
        r, c = data_id // 2, data_id % 2
        set_xlabel = r == 1
        set_ylabel = c == 0
        kwargs = dict(
            set_xlabel=set_xlabel,
            set_ylabel=set_ylabel,
            bench_id=0,
            data_id=data_id,
            hv_mode=hv_mode,
        )
        lines, labels = plot(axes[r][c], **kwargs)
    else:
        axes[-1][0].legend(handles=lines, labels=labels, ncol=(len(labels) + 1) // 2, **legend_kwargs)

    if hv_mode:
        if PLOT_CHECK_MODE:
            plt.show()
        else:
            plt.savefig("figs/hv2d-hpolib.pdf", bbox_inches='tight')
    else:
        if PLOT_CHECK_MODE:
            plt.show()
        else:
            plt.savefig("figs/eaf-hpolib.pdf", bbox_inches='tight')


def plot_hv_for_nmt(subplots_kwargs, legend_kwargs, hv_mode: bool) -> None:
    _, axes = plt.subplots(**subplots_kwargs)
    for data_id in range(3):
        set_ylabel = data_id == 0
        kwargs = dict(
            set_xlabel=True,
            set_ylabel=set_ylabel,
            bench_id=1,
            data_id=data_id,
            hv_mode=hv_mode,
        )
        lines, labels = plot(axes[data_id], **kwargs)
    else:
        axes[1].legend(handles=lines, labels=labels, ncol=(len(labels) + 1) // 2, **legend_kwargs)

    if hv_mode:
        if PLOT_CHECK_MODE:
            plt.show()
        else:
            plt.savefig("figs/hv2d-nmt.pdf", bbox_inches='tight')
    else:
        if PLOT_CHECK_MODE:
            plt.show()
        else:
            plt.savefig("figs/eaf-nmt.pdf", bbox_inches='tight')


if __name__ == "__main__":
    os.makedirs("figs/", exist_ok=True)
    subplots_kwargs = dict(
        nrows=2,
        ncols=2,
        sharex=HV_MODE,
        sharey=HV_MODE,
        figsize=(20, 10),
        gridspec_kw=dict(
            wspace=0.03 if HV_MODE else 0.09,
            hspace=0.125 if HV_MODE else 0.2,
        )
    )
    legend_kwargs = dict(
        loc='upper center',
        fontsize=20,
        bbox_to_anchor=(1.0, -0.16) if HV_MODE else (1.03, -0.16),
        fancybox=False,
        shadow=False,
    )
    plot_hv_for_hpolib(subplots_kwargs, legend_kwargs, hv_mode=HV_MODE)

    subplots_kwargs.pop("nrows")
    subplots_kwargs.update(ncols=3, figsize=(15, 3) if HV_MODE else (20, 3.5))
    legend_kwargs.update(bbox_to_anchor=(0.5, -0.3) if HV_MODE else (0.5, -0.22), fontsize=18)
    plot_hv_for_nmt(subplots_kwargs, legend_kwargs, hv_mode=HV_MODE)
