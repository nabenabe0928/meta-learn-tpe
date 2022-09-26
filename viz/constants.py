N_SAMPLES = 100
N_INIT = N_SAMPLES * 5 // 100  # N_INIT = 5
N_RUNS = 20
N_OBJ = 2
COSTS_SHAPE = (N_RUNS, N_SAMPLES, N_OBJ)
# LEVELS = [N_RUNS // 4, N_RUNS // 2, (3 * N_RUNS) // 4]
LEVELS = [N_RUNS // 2] * 3
Q, DF = [0.10, 0.15][0], [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0][2]
COLOR_LABEL_DICT = {
    # f"tpe_q={Q:.2f}_df={DF:.1f}": ("red", f"Meta-learn TPE df={DF:.1f}"),
    f"tpe_q={Q:.2f}_df={DF:.1f}": ("red", "Meta-learn TPE"),
    # f"naive_metalearn_tpe_q={Q:.2f}": ("green", "Uniform weight"),
    f"normal_tpe_q={Q:.2f}": ("blue", "MO-TPE"),
    "tstr-ehvi": ("violet", "TST-R EHVI"),
    "tstr-parego": ("violet", "TST-R ParEGO"),
    "rgpe-ehvi": ("lime", "RGPE EHVI"),
    "rgpe-parego": ("lime", "RGPE ParEGO"),
    "random": ("black", "Random"),
    "only-warmstart": ("purple", "Only warm-start"),
}
LINESTYLES_DICT = {
    "random": "dashdot",
    "only-warmstart": "dashdot",
    # f"naive_metalearn_tpe_q={Q:.2f}": "o",
    f"normal_tpe_q={Q:.2f}": "dotted",
    f"tpe_q={Q:.2f}_df={DF:.1f}": "solid",
    "tstr-ehvi": "dotted",
    "tstr-parego": "dashed",
    "rgpe-ehvi": "dotted",
    "rgpe-parego": "dashed"
}
MARKER_DICT = {
    "random": "",
    "only-warmstart": "",
    # f"naive_metalearn_tpe_q={Q:.2f}": "o",
    f"normal_tpe_q={Q:.2f}": "*",
    f"tpe_q={Q:.2f}_df={DF:.1f}": "*",
    "tstr-ehvi": "s",
    "tstr-parego": "o",
    "rgpe-ehvi": "s",
    "rgpe-parego": "o"
}
EAF_ZOOM_DICT = {
    "naval_propulsion": 2.5,
    "parkinsons_telemonitoring": 3,
    "protein_structure": 3,
    "slice_localization": 3,
}
EAF_ANCHOR_DICT = {
    "naval_propulsion": (0.995, 1.038),
    "parkinsons_telemonitoring": (1.08, 0.998),
    "protein_structure": (0.995, 1.08),
    "slice_localization": (0.995, 1.125),
}
EAF_ASPECT_DICT = {
    "naval_propulsion": 50,
    "parkinsons_telemonitoring": 100,
    "protein_structure": 600,
    "slice_localization": 400,
}
EAF_INSET_XLIM_DICT = {
    "naval_propulsion": (0, 160),
    "parkinsons_telemonitoring": (0, 100),
    "protein_structure": (40, 200),
    "slice_localization": (0, 1000),
}
EAF_INSET_YLIM_DICT = {
    "naval_propulsion": (2e-5, 1e-3),
    "parkinsons_telemonitoring": (8e-3, 2e-2),
    "protein_structure": (2.2e-1, 3e-1),
    "slice_localization": (1e-4, 2e-3),
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
OBJ_LABEL_DICT = {
    "runtime": "Runtime",
    "valid_mse": "Validation MSE",
    "decoding_time": "Runtime",
    "bleu": "BLEU",
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
