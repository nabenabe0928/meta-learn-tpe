while getopts ":s:d:" o; do
    case "${o}" in
        s) start=${OPTARG};;
        d) dx=${OPTARG};;
    esac
done

end=$((${start}+${dx}))

run_bench () {
    bench_name=${1}
    dataset_name=${2}
    for seed in `seq $start $end`
    do
        for opt_name in tpe rgpe-parego rgpe-ehvi tstr-parego tstr-ehvi
        do
            prefix="python run.py --exp_id ${seed} --opt_name ${opt_name} --bench_name ${bench_name} --dataset_name ${dataset_name}"
            if [[ "$opt_name" == "tpe" ]]
            then
                cmd="${prefix} --warmstart True --uniform_transform True"
                echo $cmd
                $cmd
                echo `date '+%y/%m/%d %H:%M:%S'`

                cmd="${prefix} --warmstart False"
                echo $cmd
                $cmd
                echo `date '+%y/%m/%d %H:%M:%S'`
            fi

            cmd="${prefix} --warmstart True"
            echo $cmd
            $cmd
            echo `date '+%y/%m/%d %H:%M:%S'`
        done
    done
}

for dataset in so_en sw_en tl_en
do
    run_bench "nmt" "${dataset}"
done

for dataset in slice_localization protein_structure naval_propulsion parkinsons_telemonitoring
do
    run_bench "hpolib" "${dataset}"
done
