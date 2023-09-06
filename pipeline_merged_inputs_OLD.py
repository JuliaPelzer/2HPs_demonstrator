import argparse
import logging
import os
import pathlib
import sys

import numpy as np
from tqdm.auto import tqdm

from domain import Domain
from heat_pump import HeatPump
from utils_2hp import set_paths

sys.path.append("/home/pelzerja/pelzerja/test_nn/1HP_NN")  # relevant for remote
sys.path.append("/home/pelzerja/Development/1HP_NN")  # relevant for local
from data_stuff.utils import load_yaml, save_yaml
from networks.models import load_model
from prepare_dataset import prepare_dataset

def pipeline(
    case: str,
    dataset_large_name: str,
    model_name_1HP: str,
    dataset_trained_model_name: str,
    input_pg: str,
    model_name_2HP: str = None,
    device: str = "cuda:0",
):
    """
    assumptions:
    - 1hp-boxes are generated already
    - network is trained
    - cell sizes of 1hp-boxes and domain are the same
    - boundaries of boxes around at least one hp is within domain
    """
    # prepare large dataset if not done yet
    (
        datasets_raw_domain_dir,
        datasets_prepared_domain_dir,
        dataset_domain_path,
        datasets_model_trained_with_path,
        model_1hp_path,
        model_2hp_path,
        name_extension,
        datasets_prepared_2hp_dir,
    ) = set_paths(
        dataset_large_name,
        model_name_1HP,
        dataset_trained_model_name,
        input_pg,
        model_name_2hp=model_name_2HP,
    )
    destination_2hp_prep = pathlib.Path(
        datasets_prepared_2hp_dir, f"{dataset_large_name}_2hp"
    )

    # load model from 1hp-NN
    model_1HP = load_model(
        {"model_choice": "unet", "in_channels": 5}, model_1hp_path, "model", device
    )

    # prepare domain dataset if not yet happened
    if not os.path.exists(dataset_domain_path):
        prepare_dataset(
            raw_data_directory=datasets_raw_domain_dir,
            datasets_path=datasets_prepared_domain_dir,
            dataset_name=dataset_large_name,
            input_variables=input_pg + "ksio",
            power2trafo=False,
            info=load_yaml(datasets_model_trained_with_path, "info"),
            name_extension=name_extension,
        )  # norm with data from dataset that NN was trained with!
    else:
        print(f"Domain {dataset_domain_path} already prepared")

    if case == "2HP apply":
        print(f"2HP-NN already prepared")
        model_2HP = load_model(
            {"model_choice": "unet", "in_channels": 5}, model_2hp_path, "model", device
        )

    # prepare 2HP dataset
    list_runs = os.listdir(os.path.join(dataset_domain_path, "Inputs"))
    for run_file in tqdm(list_runs, desc=case, total=len(list_runs)):
        run_id = f'{run_file.split(".")[0]}_'
        domain, single_hps = box_generation_2HP(
            run_file,
            run_id,
            dataset_domain_path,
            model_1HP,
            destination_2hp_prep,
            case,
            input_pg,
        )
        if case == "2HP apply":
            # model interaction of 2 heat plumes : apply 2HP-NN to 2HP dataset
            apply_2HP_NN(domain, single_hps, model_2HP, input_pg, plot=True)


def box_generation_2HP(
    run_file: str,
    run_id: int,
    dataset_domain_path: str,
    model_1HP,
    destination_2hp_prep: str,
    case: str,
    input_pg: str = "p",
    plot: bool = False,
):
    domain = Domain(dataset_domain_path, stitching_method="max", file_name=run_file)
    # generate 1hp-boxes and extract information like perm and ids etc.
    single_hps = domain.extract_hp_boxes()

    # apply learned NN to predict the heat plumes
    hp: HeatPump
    for hp in single_hps:
        hp.prediction_1HP = hp.apply_nn(model_1HP)
        if case == "1HP":  # orig. pipeline
            hp.save(dir="HP-Boxes_1HP")
        # save predicted Temp field as input for training as well
        hp.prediction_1HP = domain.reverse_norm(
            hp.prediction_1HP, property="Temperature [C]"
        )
        if case == "1HP":  # orig. pipeline
            hp.plot_1HP(domain, dir="HP-Boxes_1HP")
        domain.add_hp(hp, hp.prediction_1HP)

    if plot or case in ["1HP", "2HP prepare"]:
        domain.plot("tkio" + input_pg)

    if case in ["2HP prepare", "2HP apply"]:
        for hp in single_hps:
            domain.overwrite_boxes_prediction_1HP(
                hp
            )  # , case="separate") # CASE both inputs merged together: "merged" vs "separate": where each additional plume is an additional field
            hp.prediction_1HP = domain.norm(
                hp.prediction_1HP, property="Temperature [C]"
            )
            hp.inputs[
                domain.get_index_from_name("Original Temperature [C]")
            ] = hp.prediction_1HP.copy()
            if case == "2HP prepare":
                hp.save(run_id=run_id, dir=destination_2hp_prep)
                logging.info(f"Saved {hp.id} for run {run_id}")

        # copy info file
        if case == "2HP prepare":
            save_yaml(domain.info, path=destination_2hp_prep, name_file="info")

    return domain, single_hps


def apply_2HP_NN(
    domain: Domain,
    single_hps: list[HeatPump],
    model_2HP,
    input_pg: str = "p",
    plot: bool = False,
):
    # apply learned NN to predict the heat plumes
    domain.prediction_2HP = domain.prediction.copy()
    for hp in single_hps:
        hp.prediction_2HP = hp.apply_nn(model_2HP)
        # save predicted Temp field
        hp.prediction_2HP = domain.reverse_norm(
            hp.prediction_2HP, property="Temperature [C]"
        )
        domain.add_hp(hp, hp.prediction_2HP)
    if plot:
        domain.plot("tkio" + input_pg)


def set_paths(
    dataset_large_name: str,
    model_name_1hp: str,
    dataset_trained_model_name: str,
    input_pg: str,
    model_name_2hp: str = None,
):
    if input_pg == "g":
        name_extension = "_grad_p"
    else:
        name_extension = ""
    if os.path.exists("/scratch/sgs/pelzerja/"):
        # on remote computer: ipvsgpu1
        datasets_raw_domain_dir = "/scratch/sgs/pelzerja/datasets/2hps_demonstrator"
        datasets_prepared_domain_dir = (
            "/home/pelzerja/pelzerja/test_nn/datasets_prepared/2hps_demonstrator"
        )
        models_1hp_dir = "/home/pelzerja/pelzerja/test_nn/1HP_NN/runs"
        datasets_prepared_1hp_dir = "/home/pelzerja/pelzerja/test_nn/datasets_prepared"
    else:
        # on another computer, hopefully on lapsgs29
        datasets_raw_domain_dir = (
            "/home/pelzerja/Development/datasets/2hps_demonstrator"
        )
        datasets_prepared_domain_dir = (
            "/home/pelzerja/Development/datasets_prepared/2hps_demonstrator"
        )
        models_1hp_dir = "/home/pelzerja/Development/models/1HP_NN"
        models_2hp_dir = "/home/pelzerja/Development/models/2HP_NN"
        datasets_prepared_1hp_dir = (
            "/home/pelzerja/Development/datasets_prepared/1HP_NN"
        )
        datasets_prepared_2hp_dir = (
            "/home/pelzerja/Development/datasets_prepared/2HP_NN"
        )

    dataset_domain_path = os.path.join(
        datasets_prepared_domain_dir, dataset_large_name + name_extension
    )
    datasets_model_trained_with_path = os.path.join(
        datasets_prepared_1hp_dir, dataset_trained_model_name
    )
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
        name_extension,
        datasets_prepared_2hp_dir,
    )


def get_box_corners(pos_hp, size_hp_box, distance_hp_corner, domain_shape):
    corner_ll = pos_hp - np.array(distance_hp_corner)  # corner lower left
    corner_ur = (
        pos_hp + np.array(size_hp_box) - np.array(distance_hp_corner)
    )  # corner upper right

    assert (
        corner_ll[0] >= 0 and corner_ur[0] < domain_shape[0]
    ), f"HP BOX at {pos_hp} is with x=({corner_ll[0]}, {corner_ur[0]}) in x-direction (0, {domain_shape[0]}) not in domain"
    assert (
        corner_ll[1] >= 0 and corner_ur[1] < domain_shape[1]
    ), f"HP BOX at {pos_hp} is with y=({corner_ll[1]}, {corner_ur[1]}) in y-direction (0, {domain_shape[1]}) not in domain"

    return corner_ll, corner_ur


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_large", type=str, default="benchmark_dataset_2d_2hps_iso_perm"
    )
    parser.add_argument(
        "--model", type=str, default="current_unet_benchmark_dataset_2d_100datapoints"
    )
    parser.add_argument(
        "--dataset_boxes", type=str, default="benchmark_dataset_2d_100datapoints"
    )
    parser.add_argument("--input_pg", type=str, default="g")
    parser.add_argument("--model_2hp", type=str, default=None)
    parser.add_argument("--case", type=str)

    args = parser.parse_args()
    args.device = "cpu"

    pipeline(
        dataset_large_name=args.dataset_large,
        model_name_1HP=args.model,
        dataset_trained_model_name=args.dataset_boxes,
        case=args.case,
        input_pg=args.input_pg,
        model_name_2HP=args.model_2hp,
        device=args.device,
    )