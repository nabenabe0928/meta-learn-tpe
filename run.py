import os
import sys
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from optimizers import (
    MetaLearnGPSampler,
    RankingWeightedGaussianProcessEnsemble,
    TPEOptimizer,
    TwoStageTransferWithRanking,
)
from optimizers.convert_config_space import convert
from optimizers.warm_start_config_selector import (
    collect_metadata,
    get_result_file_path,
    save_observations,
    select_warm_start_configs,
)

from targets.base_tabularbench_api import BaseTabularBenchAPI
from targets.hpobench.api import DatasetChoices as HPOBenchChoices
from targets.hpobench.api import HPOBench
from targets.hpolib.api import DatasetChoices as HPOLibChoices
from targets.hpolib.api import HPOLib
from targets.nmt_bench.api import DatasetChoices as NMTChoices
from targets.nmt_bench.api import NMTBench


N_METADATA = 100
MAX_EVALS = 100
N_INIT = MAX_EVALS * 5 // 100  # From the TPE 2013 paper
bench_names = ["nmt", "hpolib", "hpobench"]
dataset_choices_dict = {
    bench_names[0]: NMTChoices,
    bench_names[1]: HPOLibChoices,
    bench_names[2]: HPOBenchChoices,
}
bench_dict = {
    bench_names[0]: NMTBench,
    bench_names[1]: HPOLib,
    bench_names[2]: HPOBench,
}


def get_metadata_and_warm_start_configs(
    warmstart: bool,
    metalearn: bool,
    bench: BaseTabularBenchAPI,
    bench_cls: Type[BaseTabularBenchAPI],
    dataset_choices: Union[HPOLibChoices, NMTChoices, HPOBenchChoices],
    dataset_name: str,
    seed: int,
    n_init: int,
) -> Tuple[Optional[Dict[str, Dict[str, np.ndarray]]], Optional[Dict[str, np.ndarray]]]:

    if not metalearn:
        if not warmstart:
            return None, None
        else:
            raise ValueError("no warmstart for non meta-learning methods")

    metadata = collect_metadata(
        benchmark=bench_cls,
        dataset_choices=dataset_choices,
        max_evals=N_METADATA,
        seed=seed,
        exclude=dataset_name,
    )
    if warmstart:
        warmstart_configs = select_warm_start_configs(
            metadata=metadata,
            n_configs=n_init,
            hp_names=bench.hp_names,
            obj_names=bench.obj_names,
            seed=seed,
            larger_is_better_objectives=[
                idx for idx, obj_name in enumerate(bench.obj_names) if not bench.minimize[obj_name]
            ],
        )
    else:
        random_configs = metadata[list(metadata.keys())[0]]
        # Just for Meta-learn BO methods (this is actualy random config, but not warm-starting)
        warmstart_configs = {hp_name: random_configs[hp_name][:n_init] for hp_name in bench.hp_names}

    return metadata, warmstart_configs


def format_configs(
    configs: Dict[str, np.ndarray],
    bench: BaseTabularBenchAPI,
) -> Dict[str, np.ndarray]:
    type_dict = {int: np.int32, float: np.float64}
    configs = {
        hp_name: configs[hp_name].astype(type_dict[type(bench._search_space[hp_name][0])])
        if np.issubdtype(configs[hp_name].dtype, np.number)
        else configs[hp_name]
        for hp_name in bench.hp_names
    }
    return configs


def evaluate_warmstart_configs(
    bench: BaseTabularBenchAPI,
    warmstart_configs: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    hp_names, obj_names = bench.hp_names, bench.obj_names
    n_warmstart = warmstart_configs[hp_names[0]].size
    warmstart_configs = format_configs(configs=warmstart_configs, bench=bench)
    warmstart_configs.update({obj_name: np.zeros(n_warmstart, dtype=np.float64) for obj_name in obj_names})
    for i in range(n_warmstart):
        config = {hp_name: warmstart_configs[hp_name][i] for hp_name in hp_names}
        results = obj_func(config)
        for obj_name, val in results.items():
            warmstart_configs[obj_name][i] = val

    return warmstart_configs


def optimize_by_only_warmstart(
    args: Namespace,
    bench: BaseTabularBenchAPI,
    metadata: Dict[str, Dict[str, np.ndarray]],
    warmstart_configs: Dict[str, np.ndarray],
):
    warmstart_configs = format_configs(configs=warmstart_configs, bench=bench)
    n_warmstart_configs = [warmstart_configs[key].size for key in warmstart_configs][0]
    opt = TPEOptimizer(
        obj_func=bench.objective_func,
        config_space=bench.config_space,
        objective_names=bench.obj_names,
        max_evals=n_warmstart_configs,
        minimize=bench.minimize,
        metadata=metadata,
        warmstart_configs=warmstart_configs,
        seed=args.exp_id,
    )
    opt.optimize()
    observations = opt.fetch_observations()
    n_repeats = (MAX_EVALS + n_warmstart_configs - 1) // n_warmstart_configs
    observations = {k: np.tile(v, n_repeats)[:MAX_EVALS] for k, v in observations.items()}
    return observations


def optimize_by_tpe(
    args: Namespace,
    bench: BaseTabularBenchAPI,
    metadata: Optional[Dict[str, Dict[str, np.ndarray]]],
    warmstart_configs: Optional[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    if warmstart_configs is not None:
        warmstart_configs = format_configs(configs=warmstart_configs, bench=bench)

    opt = TPEOptimizer(
        obj_func=bench.objective_func,
        config_space=bench.config_space,
        objective_names=bench.obj_names,
        max_evals=MAX_EVALS,
        minimize=bench.minimize,
        metadata=metadata,
        warmstart_configs=warmstart_configs,
        seed=args.exp_id,
        n_init=5,
        quantile=args.quantile,
        uniform_transform=args.uniform_transform,
        dim_reduction_factor=args.dim_reduction_factor,
    )
    opt.optimize()
    return opt.fetch_observations()


def convert_to_index_config(
    data: Dict[str, np.ndarray],
    search_space: Dict[str, List[Any]],
    hp_names: List[str],
) -> Dict[str, np.ndarray]:
    return {
        hp_name: np.asarray([search_space[hp_name].index(v) for v in vs])
        if np.issubdtype(vs.dtype, np.number) and hp_name in hp_names
        else vs
        for hp_name, vs in data.items()
    }


def convert_to_original_config(
    data: Dict[str, np.ndarray],
    search_space: Dict[str, List[Any]],
    hp_names: List[str],
) -> Dict[str, np.ndarray]:
    return {
        hp_name: np.asarray([search_space[hp_name][v] for v in vs])
        if np.issubdtype(vs.dtype, np.number) and hp_name in hp_names
        else vs
        for hp_name, vs in data.items()
    }


def optimize_by_bo(
    opt_name: str,
    bench: BaseTabularBenchAPI,
    metadata: Dict[str, Dict[str, np.ndarray]],
    warmstart_configs: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    metalearn_name, acq_name = opt_name.split("-")
    kwargs = convert(bench.config_space)
    kwargs.update(minimize=bench.minimize)

    hp_names = bench.hp_names
    gp_cls = RankingWeightedGaussianProcessEnsemble if metalearn_name == "rgpe" else TwoStageTransferWithRanking
    obj_func = bench.objective_func
    warmstart_configs = evaluate_warmstart_configs(bench, warmstart_configs)
    search_space = bench._search_space
    metadata = {tn: convert_to_index_config(data, search_space, hp_names) for tn, data in metadata.items()}
    warmstart_configs = convert_to_index_config(warmstart_configs, search_space, hp_names)

    gp_model = gp_cls(
        init_data=warmstart_configs,  # Need obj
        metadata=metadata,
        acq_fn_type=acq_name,
        **kwargs,
    )

    def _wrapper_func(config):
        eval_config = {k: v if isinstance(v, str) else search_space[k][v] for k, v in config.items()}
        return obj_func(eval_config)

    opt = MetaLearnGPSampler(max_evals=95, obj_func=_wrapper_func, model=gp_model, **kwargs)
    opt.optimize()
    return convert_to_original_config(data=opt.observations, search_space=search_space, hp_names=hp_names)


def get_opt_name(args: Namespace) -> str:
    opt_name = args.opt_name
    prefix = "" if args.warmstart else "no-warmstart-"
    if opt_name != "tpe":
        return prefix + opt_name
    if not args.metalearn:
        return f"normal_tpe_q={args.quantile:.2f}"
    if args.uniform_transform:
        return f"{prefix}naive_metalearn_tpe_q={args.quantile:.2f}"

    return f"{prefix}tpe_q={args.quantile:.2f}_df={args.dim_reduction_factor:.1f}"


if __name__ == "__main__":
    opt_names = ["tpe", "rgpe-parego", "rgpe-ehvi", "tstr-parego", "tstr-ehvi", "only-warmstart"]
    parser = ArgumentParser()
    parser.add_argument("--warmstart", type=str, choices=["True", "False"], required=True)
    parser.add_argument("--metalearn", type=str, choices=["True", "False"], required=True)
    parser.add_argument("--bench_name", type=str, choices=bench_names, required=True)
    dataset_choices = [c.name for c in HPOLibChoices] + [c.name for c in NMTChoices] + [c.name for c in HPOBenchChoices]
    parser.add_argument("--dataset_name", type=str, choices=dataset_choices, required=True)
    parser.add_argument("--opt_name", choices=opt_names, required=True)
    parser.add_argument("--exp_id", type=int, required=True)
    parser.add_argument("--uniform_transform", type=str, choices=["True", "False"], default="False")

    # Only for ablation study
    parser.add_argument("--quantile", type=float, default=0.1)
    parser.add_argument("--dim_reduction_factor", type=float, default=2.5)

    args = parser.parse_args()
    args.uniform_transform = eval(args.uniform_transform)
    args.warmstart, args.metalearn = eval(args.warmstart), eval(args.metalearn)
    warmstart, metalearn, bench_name, dataset_name = args.warmstart, args.metalearn, args.bench_name, args.dataset_name

    opt_name = get_opt_name(args)
    file_path = get_result_file_path(dataset_name=dataset_name, opt_name=opt_name, seed=args.exp_id)
    if os.path.exists(file_path):
        print(f"Skip: Results already exist in {file_path}\n")
        sys.exit()

    dataset_choices = dataset_choices_dict[bench_name]
    bench_cls = bench_dict[bench_name]
    bench = bench_cls(dataset=getattr(dataset_choices, dataset_name), seed=args.exp_id)

    obj_func = bench.objective_func
    config_space = bench.config_space
    only_warmstart = bool(args.opt_name == "only-warmstart")
    metadata, warmstart_configs = get_metadata_and_warm_start_configs(
        warmstart=warmstart,
        metalearn=metalearn,
        bench=bench,
        seed=args.exp_id,
        bench_cls=bench_cls,
        dataset_choices=dataset_choices,
        dataset_name=dataset_name,
        n_init=int(args.quantile * N_METADATA) * (len(dataset_choices) - 1) if only_warmstart else N_INIT,
    )
    if args.opt_name == "tpe":
        results = optimize_by_tpe(args=args, bench=bench, metadata=metadata, warmstart_configs=warmstart_configs)
    elif only_warmstart:
        results = optimize_by_only_warmstart(
            args=args, bench=bench, metadata=metadata, warmstart_configs=warmstart_configs
        )
    else:
        results = optimize_by_bo(
            opt_name=args.opt_name, bench=bench, metadata=metadata, warmstart_configs=warmstart_configs
        )

    save_observations(file_path=file_path, observations=results, include=bench.obj_names)
