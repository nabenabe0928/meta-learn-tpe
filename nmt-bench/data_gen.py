import json
import itertools as itr


METRIC_NAMES = ["bleu", "decoding_time", "perplexity", "n_updates", "gpu_memory", "n_params"]
SEARCH_SPACE = {
    "bpe": [1000, 2000, 4000, 8000, 16000, 32000],
    "n_layers": [1, 2, 4],
    "n_embed": [256, 512, 1024],
    "n_hidden": [1024, 2048],
    "n_heads": [8, 16],
    "initial_lr": [0.0003, 0.0006, 0.001],
}
CRASH_VALS = {
    "bleu": 0.0,
    "decoding_time": 1500.0,
    # "perplexity": 1e9,
    # "n_updates": 1e9,
    # "gpu_memory": 1e9,
    # "n_params": 1e9,
}


def get_data(data_name: str):
    data = json.load(open(f"{data_name}.json"))
    return data


def get_iterator():
    return itr.product(*SEARCH_SPACE.values())


def get_config_id(config) -> int:
    config_id, base = 0, 1
    for hp, (hp_name, choices) in zip(config, SEARCH_SPACE.items()):
        k = choices.index(hp)
        config_id += k * base
        base *= 6

    return config_id


def get_config_id_set(data):
    n_configs = len(data["bpe"])
    config_id_set = []
    for i in range(n_configs):
        config = (data[hp_name][i] for hp_name in SEARCH_SPACE.keys())
        config_id = get_config_id(config)
        config_id_set.append(config_id)

    return config_id_set


def pad_crash_values(data):
    config_id_set = get_config_id_set(data)
    for config in get_iterator():
        config_id = get_config_id(config)
        if config_id not in config_id_set:
            for hp_name, val in zip(SEARCH_SPACE, config):
                data[hp_name].append(val)
            for metric_name, val in CRASH_VALS.items():
                data[metric_name].append(val)

    return data


def main(data_name):
    data = get_data(data_name)
    data = pad_crash_values(data)
    with open(f"{data_name}.json", "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    for data_name in ["so-en", "sw-en", "tl-en"]:
        main(data_name)
