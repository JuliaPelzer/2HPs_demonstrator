import os
import sys
import yaml

sys.path.append("/home/pelzerja/pelzerja/test_nn/1HP_NN")  # relevant for remote
sys.path.append("/home/pelzerja/Development/1HP_NN")  # relevant for local
from data.utils import save_yaml

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

def set_paths(
    dataset_large_name: str,
    model_name_1hp: str,
    dataset_trained_model_name: str,
    # input_pg: str,
    inputs_prep: str,
    model_name_2hp: str = None,
    name_extension: str = "",
):
    # if input_pg == "g":
    #     name_extension = "_grad_p"
    # else:
    #     name_extension = ""
    
    if os.path.exists("paths.yaml"):
        with open("paths.yaml", "r") as f:
            paths = yaml.load(f, Loader=yaml.SafeLoader)
            datasets_raw_domain_dir = paths["datasets_raw_domain_dir"]
            datasets_prepared_domain_dir = paths["datasets_prepared_domain_dir"]
            models_1hp_dir = paths["models_1hp_dir"]
            models_2hp_dir = paths["models_2hp_dir"]
            datasets_prepared_1hp_dir = paths["datasets_prepared_1hp_dir"]
            datasets_prepared_2hp_dir = paths["datasets_prepared_2hp_dir"]
    else:

        if os.path.exists("/scratch/sgs/pelzerja/"):
            # on remote computer: ipvsgpu1
            datasets_raw_domain_dir = "/scratch/sgs/pelzerja/datasets/2hps_demonstrator"
            datasets_prepared_domain_dir = ("/home/pelzerja/pelzerja/test_nn/datasets_prepared/2hps_demonstrator")
            models_1hp_dir = "/home/pelzerja/pelzerja/test_nn/1HP_NN/runs/experiments"
            datasets_prepared_1hp_dir = "/home/pelzerja/pelzerja/test_nn/datasets_prepared/experiments"

    dataset_domain_path = os.path.join(datasets_prepared_domain_dir, dataset_large_name +"_"+inputs_prep + name_extension)
    datasets_model_trained_with_path = os.path.join(datasets_prepared_1hp_dir, dataset_trained_model_name)
    model_1hp_path = os.path.join(models_1hp_dir, model_name_1hp)
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
        # name_extension,
        datasets_prepared_2hp_dir,
    )
