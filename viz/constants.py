N_SAMPLES = 100
N_INIT = N_SAMPLES * 5 // 100  # N_INIT = 5
N_RUNS = 20
N_OBJ = 2
COSTS_SHAPE = (N_RUNS, N_SAMPLES, N_OBJ)
# LEVELS = [N_RUNS // 4, N_RUNS // 2, (3 * N_RUNS) // 4]
LEVELS = [N_RUNS // 2] * 3
Q, DF = [0.10, 0.15][0], [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0][3]
NO_WAMRSTART = ["no-warmstart-", ""][1]
META_LEARN_TPE = f"{NO_WAMRSTART}tpe_q={Q:.2f}_df={DF:.1f}"
NORMAL_TPE = f"normal_tpe_q={Q:.2f}"
TSTR_EHVI = "tstr-ehvi"
TSTR_PAREGO = "tstr-parego"
RGPE_EHVI = "rgpe-ehvi"
RGPE_PAREGO = "rgpe-parego"
RANDOM_SEARCH = "random"
ONLY_WS = "only-warmstart"
HPOLIB = "hpolib"
NMT = "nmt"
HPOBENCH = "hpobench"
NAVAL = "naval_propulsion"
PARKINSONS = "parkinsons_telemonitoring"
PROTEIN = "protein_structure"
SLICE = "slice_localization"
SO = "so_en"
SW = "sw_en"
TL = "tl_en"

COLOR_LABEL_DICT = {
    # f"tpe_q={Q:.2f}_df={DF:.1f}": ("red", f"Meta-learn TPE df={DF:.1f}"),
    META_LEARN_TPE: ("red", "Meta-learn TPE"),
    # f"naive_metalearn_tpe_q={Q:.2f}": ("green", "Uniform weight"),
    NORMAL_TPE: ("blue", "MO-TPE"),
    TSTR_EHVI: ("violet", "TST-R EHVI"),
    TSTR_PAREGO: ("violet", "TST-R ParEGO"),
    RGPE_EHVI: ("lime", "RGPE EHVI"),
    RGPE_PAREGO: ("lime", "RGPE ParEGO"),
    RANDOM_SEARCH: ("black", "Random"),
    ONLY_WS: ("purple", "Only warm-start"),
}
LINESTYLES_DICT = {
    META_LEARN_TPE: "solid",
    NORMAL_TPE: "dotted",
    TSTR_EHVI: "dotted",
    TSTR_PAREGO: "dashed",
    RGPE_EHVI: "dotted",
    RGPE_PAREGO: "dashed",
    RANDOM_SEARCH: "dashdot",
    ONLY_WS: "dashdot",
    # f"naive_metalearn_tpe_q={Q:.2f}": "o",
}
MARKER_DICT = {
    META_LEARN_TPE: "*",
    NORMAL_TPE: "*",
    TSTR_EHVI: "s",
    TSTR_PAREGO: "o",
    RGPE_EHVI: "s",
    RGPE_PAREGO: "o",
    RANDOM_SEARCH: "",
    ONLY_WS: "",
    # f"naive_metalearn_tpe_q={Q:.2f}": "o",
}
EAF_ZOOM_DICT = {
    NAVAL: 2.5,
    PARKINSONS: 3,
    PROTEIN: 3,
    SLICE: 3,
}
EAF_ANCHOR_DICT = {
    NAVAL: (0.995, 1.038),
    PARKINSONS: (1.08, 0.998),
    PROTEIN: (0.995, 1.08),
    SLICE: (0.995, 1.125),
}
EAF_ASPECT_DICT = {
    NAVAL: 50,
    PARKINSONS: 100,
    PROTEIN: 600,
    SLICE: 400,
}
EAF_INSET_XLIM_DICT = {
    NAVAL: (0, 160),
    PARKINSONS: (0, 100),
    PROTEIN: (40, 200),
    SLICE: (0, 1000),
}
EAF_INSET_YLIM_DICT = {
    NAVAL: (2e-5, 1e-3),
    PARKINSONS: (8e-3, 2e-2),
    PROTEIN: (2.2e-1, 3e-1),
    SLICE: (1e-4, 2e-3),
}
BENCH_NAMES = [HPOLIB, NMT, HPOBENCH]
DATASET_NAMES = {
    HPOLIB: [
        NAVAL,
        PARKINSONS,
        PROTEIN,
        SLICE,
    ],
    NMT: [
        SO,
        SW,
        TL,
    ],
    HPOBENCH: [
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
    NAVAL: "Naval Propulsion",
    PARKINSONS: "Parkinsons Telemonitoring",
    PROTEIN: "Protein Structure",
    SLICE: "Slice Localization",
    SO: "Somali to English",
    SW: "Swahili to English",
    TL: "Tagalog to English",
}

RUNTIME = "runtime"
DECODING_TIME = "decoding_time"
VALID_MSE = "valid_mse"
BLEU = "bleu"
OBJ_NAMES_DICT = {
    HPOLIB: [RUNTIME, VALID_MSE],
    NMT: [DECODING_TIME, BLEU],
    HPOBENCH: ["precision", "bal_acc"]
}
OBJ_LABEL_DICT = {
    RUNTIME: "Runtime",
    VALID_MSE: "Validation MSE",
    DECODING_TIME: "Runtime",
    BLEU: "BLEU",
}
LARGER_IS_BETTER_DICT = {
    HPOLIB: None,
    NMT: [1],
    HPOBENCH: [0, 1],
}
LOGSCALE_DICT = {
    HPOLIB: [1],
    NMT: None,
    HPOBENCH: None,
}
TICK_PARAMS = dict(
    labelleft=False,
    labelbottom=False,
    left=False,
    bottom=False,
)
