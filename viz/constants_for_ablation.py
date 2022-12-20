import matplotlib.pyplot as plt
import numpy as np


CMAP = plt.get_cmap("rainbow")


N_SAMPLES = 100
N_INIT = N_SAMPLES * 5 // 100  # N_INIT = 5
N_RUNS = 20
N_OBJ = 2
COSTS_SHAPE = (N_RUNS, N_SAMPLES, N_OBJ)
# LEVELS = [N_RUNS // 4, N_RUNS // 2, (3 * N_RUNS) // 4]
LEVELS = [N_RUNS // 2] * 3
Q, DF = [0.10, 0.15][0], [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
COLORS = [CMAP(v) for v in np.linspace(0, 1, len(DF))]
NO_WAMRSTART = ["no-warmstart-", ""][1]
META_LEARN_TPE = [f"{NO_WAMRSTART}tpe_q={Q:.2f}_df={df:.1f}" for df in DF]
NORMAL_TPE = f"normal_tpe_q={Q:.2f}"
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
    name: (color, f"$\\eta = {name.split('df=')[-1]}$")
    for color, name in zip(COLORS, META_LEARN_TPE)
}
LINESTYLES_DICT = {
    name: "solid"
    for name in META_LEARN_TPE
}
MARKER_DICT = {
    name: "*"
    for name in META_LEARN_TPE
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
