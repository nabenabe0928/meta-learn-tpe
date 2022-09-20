while getopts ":s:d:" o; do
    case "${o}" in
        s) start=${OPTARG};;
        d) dx=${OPTARG};;
    esac
done

end=$((${start}+${dx}))

run_tpe () {
    prefix=${1}

    for quantile in 0.10 0.15
    do
        cmd="${prefix} --warmstart True --uniform_transform True --quantile ${quantile}"
        echo $cmd
        $cmd
        echo `date '+%y/%m/%d %H:%M:%S'`

        cmd="${prefix} --warmstart False --quantile ${quantile}"
        echo $cmd
        $cmd
        echo `date '+%y/%m/%d %H:%M:%S'`

        for df in `seq 2 4`
        do
            cmd="${prefix} --warmstart True --dim_reduction_factor ${df} --quantile ${quantile}"
            echo $cmd
            $cmd
            echo `date '+%y/%m/%d %H:%M:%S'`
        done
    done
}

run_bench () {
    seed=${1}
    bench_name=${2}
    dataset_name=${3}
    for opt_name in tpe rgpe-parego rgpe-ehvi tstr-parego tstr-ehvi
    do
        prefix="python run.py --exp_id ${seed} --opt_name ${opt_name} --bench_name ${bench_name} --dataset_name ${dataset_name}"
        if [[ "$opt_name" == "tpe" ]]
        then
            run_tpe "${prefix}"
        else
            cmd="${prefix} --warmstart True"
            echo $cmd
            $cmd
            echo `date '+%y/%m/%d %H:%M:%S'`
        fi
    done
}

for seed in `seq $start $end`
do
    for dataset in so_en sw_en tl_en
    do
        run_bench ${seed} "nmt" "${dataset}"
    done

    for dataset in slice_localization protein_structure naval_propulsion parkinsons_telemonitoring
    do
        run_bench ${seed} "hpolib" "${dataset}"
    done

    for dataset in credit_g vehicle kc1 phoneme blood_transfusion australian car segment
    do
        run_bench ${seed} "hpobench" "${dataset}"
    done
done
