import argparse
import logging
import os
import pathlib
import sys
import time

import numpy as np
import yaml
from tqdm.auto import tqdm

from domain import Domain
from heat_pump import HeatPump
from utils_2hp import save_config_of_separate_inputs, set_paths

sys.path.append("/home/pelzerja/pelzerja/test_nn/1HP_NN")  # relevant for remote
sys.path.append("/home/pelzerja/Development/1HP_NN")  # relevant for local
from data_stuff.utils import SettingsPrepare, load_yaml, save_yaml
from networks.models import load_model
from prepare_dataset import prepare_dataset
from utils.utils import beep


def prepare_separate_inputs_for_2HPNN(
    dataset_large_name: str,
    preparation_case: str,
    device: str = "cuda:0",
):
    """
    assumptions:
    - 1hp-boxes are generated already
    - network is trained
    - cell sizes of 1hp-boxes and domain are the same
    - boundaries of boxes around at least one hp is within domain
    """
    timestamp_begin = time.ctime()
    time_begin = time.perf_counter()

    # prepare large dataset if not done yet
    (datasets_raw_domain_dir, datasets_prepared_domain_dir, dataset_domain_path, datasets_model_trained_with_path, model_1hp_path, _, datasets_prepared_2hp_dir, inputs_prep
     ) = set_paths(dataset_large_name, preparation_case)
    destination_2hp_prep = pathlib.Path(datasets_prepared_2hp_dir, f"{dataset_large_name}_2hp_{preparation_case}")
    destination_2hp_prep.mkdir(parents=True, exist_ok=True)

    # load model from 1hp-NN
    model_1HP = load_model(
        {"model_choice": "unet", "in_channels": len(inputs_prep)}, model_1hp_path, "model", device
    )

    time_start_prep_domain = time.perf_counter()
    # prepare domain dataset if not yet happened
    if not os.path.exists(dataset_domain_path):
        args = SettingsPrepare(
            raw_dir=datasets_raw_domain_dir,
            datasets_dir=datasets_prepared_domain_dir,
            dataset_name=dataset_large_name,
            inputs_prep=inputs_prep,
            # name_extension=name_extension,
        )
        prepare_dataset(args,
            power2trafo=False,
            info=load_yaml(datasets_model_trained_with_path, "info"),
        )  # norm with data from dataset that NN was trained with!
        print(f"Domain {dataset_domain_path} prepared")
    else:
        print(f"Domain {dataset_domain_path} already prepared")
    
    time_start_prep_2hp = time.perf_counter()
    avg_time_inference_1hp = 0
    # prepare 2HP dataset
    list_runs = os.listdir(os.path.join(dataset_domain_path, "Inputs"))
    for run_file in tqdm(list_runs, desc="2HP prepare", total=len(list_runs)):
        run_id = f'{run_file.split(".")[0]}_'
        domain = Domain(dataset_domain_path, stitching_method="max", file_name=run_file)
        # generate 1hp-boxes and extract information like perm and ids etc.
        if domain.skip_datapoint:
            logging.warning(f"Skipping {run_id}")
            continue

        single_hps = domain.extract_hp_boxes()
        # apply learned NN to predict the heat plumes
        hp: HeatPump
        for hp in single_hps:
            time_start_run_1hp = time.perf_counter()
            hp.prediction_1HP = hp.apply_nn(model_1HP)
            avg_time_inference_1hp += time.perf_counter() - time_start_run_1hp
            hp.prediction_1HP = domain.reverse_norm(
                hp.prediction_1HP, property="Temperature [C]"
            )

        for hp in single_hps:
            hp.get_inputs_for_2hp_nn(single_hps)
            # domain.add_hp(hp, hp.prediction_1HP)

        for hp in single_hps:
            hp.prediction_1HP = domain.norm(
                hp.prediction_1HP, property="Temperature [C]"
            )
            hp.interim_outputs = domain.norm(
                hp.interim_outputs, property="Temperature [C]"
            )
            hp.save(
                run_id=run_id,
                dir=destination_2hp_prep,
                inputs_all=np.array([hp.prediction_1HP, hp.interim_outputs]),
            )
            # hp.plot(dir=destination_2hp_prep, data_to_plot=np.array([hp.prediction_1HP, hp.interim_outputs]), names=np.array(["prediction_1HP", "other HPs temperature field"]))
            logging.info(f"Saved {hp.id} for run {run_id}")
        # domain.plot("goksit")
    time_end = time.perf_counter()
    avg_inference_times = avg_time_inference_1hp / len(list_runs)

    # save infos of info file about separated (only 2!) inputs
    save_config_of_separate_inputs(
        domain.info, path=destination_2hp_prep, name_file="info"
    )
    # save command line arguments
    cla = {
        "dataset_large_name": dataset_large_name,
        "preparation_case": preparation_case,
    }
    save_yaml(cla, path=destination_2hp_prep, name_file="command_line_args")

    # save measurements
    with open(os.path.join(os.getcwd(), "runs", destination_2hp_prep, f"measurements.yaml"), "w") as f:
        f.write(f"timestamp of beginning: {timestamp_begin}\n")
        f.write(f"timestamp of end: {time.ctime()}\n")
        f.write(f"model 1HP: {model_1hp_path}\n")
        f.write(f"input params: {inputs_prep}\n")
        f.write(f"separate inputs: {True}\n")
        f.write(f"dataset prepared location: {datasets_prepared_domain_dir}\n")
        f.write(f"dataset name: {datasets_model_trained_with_path}\n")
        f.write(f"dataset large name: {dataset_large_name}\n")
        f.write(f"name_destination_folder: {destination_2hp_prep}\n")
        f.write(f"avg inference times for 1HP-NN in seconds: {avg_inference_times}\n")
        f.write(f"device: {device}\n")
        f.write(f"duration of preparing domain in seconds: {(time_start_prep_2hp-time_start_prep_domain)}\n")
        f.write(f"duration of preparing 2HP in seconds: {(time_end-time_start_prep_2hp)}\n")
        f.write(f"duration of preparing 2HP /run in seconds: {(time_end-time_start_prep_2hp)/len(list_runs)}\n")
        f.write(f"duration of whole process in seconds: {(time_end-time_begin)}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation_case", type=str, default="gksi_100dp")
    parser.add_argument("--dataset_large", type=str, default="benchmark_dataset_2d_2hps_iso_perm")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    assert args.preparation_case in ["gksi_100dp", "ogksi_1000dp", "gksi_1000dp", "pksi_100dp", "pksi_1000dp"], "preparation_case must be one of ['gksi_100dp', 'gksi_1000dp', 'pksi_100dp', 'pksi_1000dp']"

    prepare_separate_inputs_for_2HPNN(
        dataset_large_name=args.dataset_large,
        preparation_case=args.preparation_case,
        device=args.device,
    )

    # beep()
