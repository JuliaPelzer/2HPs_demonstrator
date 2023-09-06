import os
import pathlib
import sys
from typing import List

import yaml

sys.path.append("/home/pelzerja/pelzerja/test_nn/1HP_NN")  # relevant for remote
sys.path.append("/home/pelzerja/Development/1HP_NN")  # relevant for local
from data_stuff.utils import save_yaml


def save_config_of_separate_inputs(domain_info, path, name_file="info"):
    # ATTENTION! Lots of hardcoding
    temperature_info = domain_info["Labels"]["Temperature [C]"]
    shortened_input = {
        "Temperature prediction (1st HP) [C]": temperature_info.copy(),
        "Temperature prediction (other HPs) [C]": temperature_info.copy(),
    }
    shortened_input["Temperature prediction (1st HP) [C]"]["index"] = 0
    shortened_input["Temperature prediction (other HPs) [C]"]["index"] = 1
    shortened_infos = {
        "Inputs": shortened_input,
        "Labels": domain_info["Labels"],
        "CellsNumber": domain_info["CellsNumber"],
        "CellsNumberPrior": domain_info["CellsNumberPrior"],
        "CellsSize": domain_info["CellsSize"],
    }
    save_yaml(shortened_infos, path, name_file)

def save_config_of_merged_inputs(separate_info, path, name_file="info"):
    # ATTENTION! Lots of hardcoding
    temperature_info = separate_info["Labels"]["Temperature [C]"]
    shortened_input = {
        "Temperature prediction (merged) [C]": temperature_info.copy(),
    }
    shortened_input["Temperature prediction (merged) [C]"]["index"] = 0
    shortened_infos = {
        "Inputs": shortened_input,
        "Labels": separate_info["Labels"],
        "CellsNumber": separate_info["CellsNumber"],
        "CellsNumberPrior": separate_info["CellsNumberPrior"],
        "CellsSize": separate_info["CellsSize"],
    }
    save_yaml(shortened_infos, path, name_file)

def set_paths(
    dataset_large_name: str,
    preparation_case: str,
    model_name_2hp: str = None,
    name_extension: str = "",
):
    if os.path.exists("paths.yaml"):
        with open("paths.yaml", "r") as f:
            paths = yaml.load(f, Loader=yaml.SafeLoader)
            datasets_raw_domain_dir = paths["datasets_raw_domain_dir"]
            datasets_prepared_domain_dir = paths["datasets_prepared_domain_dir"]
            prepared_1hp_dir = paths["prepared_1hp_dir"]
            models_2hp_dir = paths["models_2hp_dir"]
            datasets_prepared_2hp_dir = paths["datasets_prepared_2hp_dir"]
    else:
        # error
        raise Exception("paths.yaml does not exist")
    
    # get model name, get dataset, get inputs
    prepared_1hp_dir = pathlib.Path(prepared_1hp_dir) / preparation_case
    for path in prepared_1hp_dir.iterdir():
        if path.is_dir():
            if "current" in path.name:
                model_1hp_path = prepared_1hp_dir / path.name
            elif "dataset" in path.name:
                datasets_model_trained_with_path = prepared_1hp_dir / path.name

    inputs_prep = str(preparation_case).split("_")[0]
    
    dataset_domain_path = pathlib.Path(datasets_prepared_domain_dir) / f"{dataset_large_name}_{inputs_prep}{name_extension}"
    model_2hp_path = None
    if model_name_2hp is not None:
        model_2hp_path = os.path.join(models_2hp_dir, model_name_2hp)

    return (
        datasets_raw_domain_dir,
        datasets_prepared_domain_dir,
        dataset_domain_path,
        datasets_model_trained_with_path,
        model_1hp_path,
        model_2hp_path,
        datasets_prepared_2hp_dir,
        inputs_prep,
    )

def check_all_datasets_prepared(paths: List):
    # check if all datasets required are prepared ( domain and 2hp-nn dataset )
    for path in paths:
        if not os.path.exists(path):
            # error
            raise ValueError(f"Dataset {path} not prepared")
        else:
            print(f"Dataset {path} prepared")
