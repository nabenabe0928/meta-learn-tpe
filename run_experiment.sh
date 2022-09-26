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
        cmd="${prefix} --warmstart False --metalearn False --quantile ${quantile}"
        echo $cmd
        $cmd
        echo `date '+%y/%m/%d %H:%M:%S'`

        for warmstart in True False
        do
            cmd="${prefix} --warmstart ${warmstart} --metalearn True --uniform_transform True --quantile ${quantile}"
            echo $cmd
            $cmd
            echo `date '+%y/%m/%d %H:%M:%S'`

            for df in 1.5 2 2.5 3 3.5 4 4.5 5
            do
                cmd="${prefix} --warmstart ${warmstart} --metalearn True --dim_reduction_factor ${df} --quantile ${quantile}"
                echo $cmd
                $cmd
                echo `date '+%y/%m/%d %H:%M:%S'`
            done
        done
    done
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
            for warmstart in True False
            do
                cmd="${prefix} --warmstart ${warmstart} --metalearn True"
                echo $cmd
                $cmd
                echo `date '+%y/%m/%d %H:%M:%S'`
            done
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
