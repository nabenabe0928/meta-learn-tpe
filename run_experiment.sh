while getopts ":s:d:" o; do
    case "${o}" in
        s) start=${OPTARG};;
        d) dx=${OPTARG};;
    esac
done

end=$((${start}+${dx}))

run_tpe () {
    prefix=${1}
    quantile=0.10

    # Normal MO-TPE
    cmd="${prefix} --warmstart False --metalearn False --quantile ${quantile}"
    echo $cmd
    $cmd
    echo `date '+%y/%m/%d %H:%M:%S'`

    # Meta-learning MO-TPE
    cmd="${prefix} --warmstart ${warmstart} --metalearn True --dim_reduction_factor 2.5 --quantile ${quantile}"
    echo $cmd
    $cmd
    echo `date '+%y/%m/%d %H:%M:%S'`
}

run_bench () {
    seed=${1}
    bench_name=${2}
    dataset_name=${3}
    for opt_name in tpe only-warmstart rgpe-parego rgpe-ehvi tstr-parego tstr-ehvi
    do
        prefix="python run.py --exp_id ${seed} --opt_name ${opt_name} --bench_name ${bench_name} --dataset_name ${dataset_name}"
        if [[ "$opt_name" == "tpe" ]]
        then
            run_tpe "${prefix}"
        else
            cmd="${prefix} --warmstart True --metalearn True"
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
done
