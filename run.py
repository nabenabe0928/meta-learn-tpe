from argparse import ArgumentParser, Namespace
from typing import Dict, Optional, Tuple, Type, Union

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
    select_warm_start_configs
)

from targets.base_tabularbench_api import BaseTabularBenchAPI
from targets.hpolib.api import DatasetChoices as HPOChoices
from targets.hpolib.api import HPOBench
from targets.nmt_bench.api import DatasetChoices as NMTChoices
from targets.nmt_bench.api import NMTBench


bench_names = ["nmt", "hpolib"]
dataset_choices_dict = {
    bench_names[0]: NMTChoices,
    bench_names[1]: HPOChoices,
}
bench_dict = {
    bench_names[0]: NMTBench,
    bench_names[1]: HPOBench,
}


def get_metadata_and_warm_start_configs(
    warmstart: bool,
    bench: BaseTabularBenchAPI,
    bench_cls: Type[BaseTabularBenchAPI],
    dataset_choices: Union[HPOChoices, NMTChoices],
    dataset_name: str,
    seed: int,
) -> Tuple[Optional[Dict[str, Dict[str, np.ndarray]]], Optional[Dict[str, np.ndarray]]]:
    if warmstart:
        # TODO: Save the metadata for the reuse
        metadata = collect_metadata(
            benchmark=bench_cls,
            dataset_choices=dataset_choices,
            max_evals=100,
            seed=seed,
            exclude=dataset_name,
        )
        warmstart_configs = select_warm_start_configs(
            metadata=metadata,
            n_configs=5,
            hp_names=bench.hp_names,
            obj_names=bench.obj_names,
        )
        return metadata, warmstart_configs
    else:
        return None, None


def evaluate_warmstart_configs(
    bench: BaseTabularBenchAPI,
    warmstart_configs: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    hp_names, obj_names = bench.hp_names, bench.obj_names
    n_warmstart = warmstart_configs[hp_names[0]].size
    warmstart_configs.update({obj_name: np.zeros(n_warmstart, dtype=np.float64) for obj_name in obj_names})
    for i in range(n_warmstart):
        config = {hp_name: warmstart_configs[hp_name][i] for hp_name in hp_names}
        results = obj_func(config)
        for obj_name, val in results.items():
            warmstart_configs[obj_name][i] = val

    return warmstart_configs


def optimize_by_tpe(
    args: Namespace,
    bench: BaseTabularBenchAPI,
    metadata: Optional[Dict[str, Dict[str, np.ndarray]]],
    warmstart_configs: Optional[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    opt = TPEOptimizer(
        obj_func=bench.obj_func,
        config_space=bench.config_space,
        objective_names=bench.obj_names,
        max_evals=100,
        minimize=bench.minimize,
        metadata=metadata,
        warmstart_configs=warmstart_configs,
        seed=args.exp_id,
        quantile=args.quantile,
        uniform_transform=args.uniform_transform,
        dim_reduction_factor=args.dim_reduction_factor,
    )
    opt.optimize()
    return opt.fetch_observations()


def optimize_by_bo(
    opt_name: str,
    bench: BaseTabularBenchAPI,
    metadata: Dict[str, Dict[str, np.ndarray]],
    warmstart_configs: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    metalearn_name, acq_name = opt_name.split("-")
    kwargs = convert(bench.config_space)
    kwargs.update(minimize=bench.minimize)

    gp_cls = RankingWeightedGaussianProcessEnsemble if metalearn_name == "rgpe" else TwoStageTransferWithRanking
    obj_func = bench.objective_func
    warmstart_configs = evaluate_warmstart_configs(bench, warmstart_configs)

    gp_model = gp_cls(
        init_data=warmstart_configs,  # Need obj
        metadata=metadata,
        acq_fn_type=acq_name,
        **kwargs,
    )
    opt = MetaLearnGPSampler(max_evals=100, obj_func=obj_func, model=gp_model, **kwargs)
    opt.optimize()
    return opt.observations


def get_opt_name(args: Namespace) -> str:
    opt_name = args.opt_name
    if args != "tpe":
        return opt_name
    if not args.warmstart:
        return "normal_tpe"
    if args.uniform_transform:
        return "naive_metalearn_tpe"

    return f"tpe_q={args.quantile:.2f}_df={args.dim_reduction_factor:.0f}"


if __name__ == "__main__":
    opt_names = ["tpe", "rgpe-parego", "rgpe-ehvi", "tstr-parego", "tstr-ehvi"]
    parser = ArgumentParser()
    parser.add_argument("--warmstart", type=str, choices=["True", "False"])
    parser.add_argument("--bench_name", type=str, choices=bench_names)
    parser.add_argument("--dataset_name", type=str, choices=[c.name for c in HPOChoices] + [c.name for c in NMTChoices])
    parser.add_argument("--opt_name", choices=opt_names)
    parser.add_argument("--exp_id", type=int)
    parser.add_argument("--uniform_transform", type=str, choices=["True", "False"], default="True")

    # Only for ablation study
    parser.add_argument("--quantile", type=float, default=0.1)
    parser.add_argument("--dim_reduction_factor", type=float, default=5.0)

    args = parser.parse_args()
    args.uniform_transform, args.warmstart = eval(args.uniform_transform), eval(args.warmstart)
    warmstart, bench_name, dataset_name = args.warmstart, args.bench_name, args.dataset_name

    dataset_choices = dataset_choices_dict[bench_name]
    bench_cls = bench_dict[bench_name]
    bench = bench_cls(dataset=getattr(dataset_choices, dataset_name))

    obj_func = bench.objective_func
    config_space = bench.config_space
    metadata, warmstart_configs = get_metadata_and_warm_start_configs(
        warmstart=warmstart,
        bench=bench,
        seed=args.exp_id,
        bench_cls=bench_cls,
        dataset_choices=dataset_choices,
        dataset_name=dataset_name,
    )
    if args.opt_name == "tpe":
        results = optimize_by_tpe(args=args, bench=bench, metadata=metadata, warmstart_configs=warmstart_configs)
    else:
        results = optimize_by_bo(
            opt_name=args.opt_name, bench=bench, metadata=metadata, warmstart_configs=warmstart_configs
        )

    opt_name = get_opt_name(args)
    file_path = get_result_file_path(dataset_name=dataset_name, opt_name=opt_name, seed=args.exp_id)
    save_observations(file_path=file_path, observations=results, include=bench.obj_names)
