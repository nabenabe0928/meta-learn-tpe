while getopts ":s:d:" o; do
    case "${o}" in
        s) start=${OPTARG};;
        d) dx=${OPTARG};;
    esac
done

end=$((${start}+${dx}))

run_bench () {
    seed=${1}
    bench_name=${2}
    dataset_name=${3}
    for opt_name in rgpe-parego rgpe-ehvi tstr-parego tstr-ehvi tpe
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
