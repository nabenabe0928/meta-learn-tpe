N_SAMPLES = 100
N_INIT = N_SAMPLES * 5 // 100
N_RUNS = 20
N_OBJ = 2
COSTS_SHAPE = (N_RUNS, N_SAMPLES, N_OBJ)
LEVELS = [N_RUNS // 4, N_RUNS // 2, (3 * N_RUNS) // 4]
# LEVELS = [N_RUNS // 2] * 3
HV_MODE = True
Q, DF = [0.10, 0.15][0], [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0][2]
WARMSTART_OPT = f"tpe_q={Q:.2f}_df={DF:.1f}"
COLOR_LABEL_DICT = {
    "random": ("olive", "Random"),
    # f"naive_metalearn_tpe_q={Q:.2f}": ("green", "Uniform weight"),
    f"normal_tpe_q={Q:.2f}": ("blue", "MOTPE"),
    # f"tpe_q={Q:.2f}_df={DF:.1f}": ("red", f"Meta-learn TPE df={DF:.1f}"),
    f"tpe_q={Q:.2f}_df={DF:.1f}": ("red", "Meta-learn TPE"),
    "warmstart_config": ("black", "Warmstart configs"),
    "tstr-ehvi": ("cyan", "TST-R EHVI"),
    "tstr-parego": ("magenta", "TST-R ParEGO"),
    "rgpe-ehvi": ("purple", "RGPE EHVI"),
    "rgpe-parego": ("brown", "RGPE ParEGO"),
}
MARKER_DICT = {
    "random": "",
    # f"naive_metalearn_tpe_q={Q:.2f}": "o",
    f"normal_tpe_q={Q:.2f}": "o",
    f"tpe_q={Q:.2f}_df={DF:.1f}": "*",
    "warmstart_config": "",
    "tstr-ehvi": "v",
    "tstr-parego": "^",
    "rgpe-ehvi": "1",
    "rgpe-parego": "2"
}
BENCH_NAMES = ["hpolib", "nmt", "hpobench"]
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
    "hpobench": [
        "credit_g",
        "vehicle",
        "kc1",
        "phoneme",
        "blood_transfusion",
        "australian",
        "car",
        "segment",
    ]
}
NAME_DICT = {
    "naval_propulsion": "Naval Propulsion",
    "parkinsons_telemonitoring": "Parkinsons Telemonitoring",
    "protein_structure": "Protein Structure",
    "slice_localization": "Slice Localization",
    "so_en": "Somali to English",
    "sw_en": "Swahili to English",
    "tl_en": "Tagalog to English",
}
OBJ_NAMES_DICT = {
    "hpolib": ["runtime", "valid_mse"],
    "nmt": ["decoding_time", "bleu"],
    "hpobench": ["precision", "bal_acc"]
}
LARGER_IS_BETTER_DICT = {
    "hpolib": None,
    "nmt": [1],
    "hpobench": [0, 1],
}
LOGSCALE_DICT = {
    "hpolib": [1],
    "nmt": None,
    "hpobench": None,
}
TICK_PARAMS = dict(
    labelleft=False,
    labelbottom=False,
    left=False,
    bottom=False,
)
