import argparse
import logging
import os
import pathlib
import shutil
import sys
from math import cos, sin

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor, load, save, squeeze, unsqueeze
from tqdm.auto import tqdm

sys.path.append("/home/pelzerja/pelzerja/test_nn/1HP_NN")  # relevant for remote
sys.path.append("/home/pelzerja/Development/1HP_NN")  # relevant for local
from data.transforms import SignedDistanceTransform
from data.utils import load_yaml, save_yaml
from networks.models import load_model
from prepare_dataset import expand_property_names, prepare_dataset
from utils.utils import beep
from utils.visualize_data import _aligned_colorbar


class Domain:
    def __init__(
        self, info_path: str, stitching_method: str = "max", file_name: str = "RUN_0.pt"
    ):
        self.info = load_yaml(info_path, "info")
        self.size: tuple[int, int] = [
            self.info["CellsNumber"][0],
            self.info["CellsNumber"][1],
        ]  # (x, y), cell-ids
        self.background_temperature: float = 10.6
        self.inputs: np.ndarray = self.load_datapoint(
            info_path, case="Inputs", file_name=file_name
        )
        self.label: np.ndarray = self.load_datapoint(
            info_path, case="Labels", file_name=file_name
        )
        self.prediction: np.ndarray = np.ones(self.size) * self.background_temperature
        self.prediction_2HP: np.ndarray = (
            np.ones(self.size) * self.background_temperature
        )
        self.stitching: Stitching = Stitching(
            stitching_method, self.background_temperature
        )
        self.normed: bool = True
        self.file_name: str = file_name
        if (
            self.get_input_field_from_name("Permeability X [m^2]").max() > 1
            or self.get_input_field_from_name("Permeability X [m^2]").min() < 0
        ):
            origin_2hp_prep = "/home/pelzerja/Development/datasets_prepared/2hps_demonstrator/dataset_2hps_1fixed_1000dp_grad_p"
            # TODO rm absolute path
            shutil.move(
                os.path.join(origin_2hp_prep, "Inputs", file_name),
                os.path.join(
                    origin_2hp_prep,
                    "broken",
                    "Inputs",
                    f"{file_name.split('.')[0]}_perm_outside_0_1.pt",
                ),
            )
            shutil.move(
                os.path.join(origin_2hp_prep, "Labels", file_name),
                os.path.join(origin_2hp_prep, "broken", "Labels", file_name),
            )
            beep()
        assert (
            self.get_input_field_from_name("Permeability X [m^2]").max() <= 1
        ), f"Max of permeability X [m^2] not < 1 but {self.get_input_field_from_name('Permeability X [m^2]').max()} for {file_name}"
        assert (
            self.get_input_field_from_name("Permeability X [m^2]").min() >= 0
        ), f"Min of permeability X [m^2] not > 0 but {self.get_input_field_from_name('Permeability X [m^2]').min()} for {file_name}"
        # TODO : wenn perm/pressure nicht mehr konstant sind, muss dies zu den HP-Boxen verschoben werden
        try:
            p_related_name = "Pressure Gradient [-]"
            p_related_field = self.get_input_field_from_name(p_related_name)
        except:
            p_related_name = "Liquid Pressure [Pa]"
            p_related_field = self.get_input_field_from_name(p_related_name)
        logging.info(
            f"{p_related_name} in range ({p_related_field.max()}, {p_related_field.min()})"
        )

        if p_related_field.max() > 1 or p_related_field.min() < 0:
            origin_2hp_prep = "/home/pelzerja/Development/datasets_prepared/2hps_demonstrator/dataset_2hps_1fixed_1000dp_grad_p"
            # TODO rm absolute path
            shutil.move(
                os.path.join(origin_2hp_prep, "Inputs", file_name),
                os.path.join(
                    origin_2hp_prep,
                    "broken",
                    "Inputs",
                    f"{file_name.split('.')[0]}_p_outside_0_1.pt",
                ),
            )
            shutil.move(
                os.path.join(origin_2hp_prep, "Labels", file_name),
                os.path.join(origin_2hp_prep, "broken", "Labels", file_name),
            )
            beep()
        assert (
            p_related_field.max() <= 1 and p_related_field.min() >= 0
        ), f"{p_related_name} not in range (0,1) but {p_related_field.max(), p_related_field.min()}"

    def load_datapoint(
        self, dataset_domain_path: str, case: str = "Inputs", file_name="RUN_0.pt"
    ):
        # load dataset of large domain
        file_path = os.path.join(dataset_domain_path, case, file_name)
        data = load(file_path).detach().numpy()
        return data

    def get_index_from_name(self, name: str):
        return self.info["Inputs"][name]["index"]

    def get_name_from_index(self, index: int):
        for property, values in self.info["Inputs"].items():
            if values["index"] == index:
                return property

    def get_input_field_from_name(self, name: str):
        field_idx = self.get_index_from_name(name)
        field = self.inputs[field_idx, :, :]
        return field

    def norm(self, data: np.ndarray, property: str = "Temperature [C]"):
        norm_fct, max_val, min_val, mean_val, std_val = self.get_norm_info(property)

        if norm_fct == "Rescale":
            out_min, out_max = (
                0,
                1,
            )  # TODO Achtung! Hardcoded, values same as in transforms.NormalizeTransform.out_min/max
            delta = max_val - min_val
            data = (data - min_val) / delta * (out_max - out_min) + out_min
        elif norm_fct == "Standardize":
            data = (data - mean_val) / std_val
        elif norm_fct is None:
            pass
        else:
            raise ValueError(f"Normalization type '{self.norm['Norm']}' not recognized")
        return data

    def reverse_norm(self, data: np.ndarray, property: str = "Temperature [C]"):
        norm_fct, max_val, min_val, mean_val, std_val = self.get_norm_info(property)

        if norm_fct == "Rescale":
            out_min, out_max = (
                0,
                1,
            )  # TODO Achtung! Hardcoded, values same as in transforms.NormalizeTransform.out_min/max
            delta = max_val - min_val
            data = (data - out_min) / (out_max - out_min) * delta + min_val
        elif norm_fct == "Standardize":
            data = data * std_val + mean_val
        elif norm_fct is None:
            pass
        else:
            raise ValueError(
                f"Normalization type '{self.norm_fct['Norm']}' not recognized"
            )
        return data

    def get_norm_info(self, property: str = "Temperature [C]"):
        try:
            norm_fct = self.info["Inputs"][property]["norm"]
            max_val = self.info["Inputs"][property]["max"]
            min_val = self.info["Inputs"][property]["min"]
            mean_val = self.info["Inputs"][property]["mean"]
            std_val = self.info["Inputs"][property]["std"]
        except:
            norm_fct = self.info["Labels"][property]["norm"]
            max_val = self.info["Labels"][property]["max"]
            min_val = self.info["Labels"][property]["min"]
            mean_val = self.info["Labels"][property]["mean"]
            std_val = self.info["Labels"][property]["std"]
        return norm_fct, max_val, min_val, mean_val, std_val

    def extract_hp_boxes(self) -> list:
        # TODO decide: get hp_boxes based on grad_p or based on v or get squared boxes around hp
        material_ids = self.get_input_field_from_name("Material ID")
        size_hp_box = [
            self.info["CellsNumberPrior"][0],
            self.info["CellsNumberPrior"][1],
        ]
        distance_hp_corner = [
            self.info["PositionHPPrior"][0],
            self.info["PositionHPPrior"][1],
        ]
        hp_boxes = []
        pos_hps = np.array(np.where(material_ids == np.max(material_ids))).T
        for idx in range(len(pos_hps)):
            try:
                pos_hp = pos_hps[idx]
                corner_ll, corner_ur = get_box_corners(
                    pos_hp,
                    size_hp_box,
                    distance_hp_corner,
                    self.inputs.shape[1:],
                    run_name=self.file_name,
                )
                tmp_input = self.inputs[
                    :, corner_ll[0] : corner_ur[0], corner_ll[1] : corner_ur[1]
                ].copy()
                tmp_label = self.label[
                    :, corner_ll[0] : corner_ur[0], corner_ll[1] : corner_ur[1]
                ].copy()

                tmp_mat_ids = np.array(np.where(tmp_input == np.max(material_ids))).T
                if len(tmp_mat_ids) > 1:
                    for i in range(len(tmp_mat_ids)):
                        tmp_pos = tmp_mat_ids[i]
                        if (tmp_pos[1:2] != distance_hp_corner).all():
                            tmp_input[tmp_pos[0], tmp_pos[1], tmp_pos[2]] = 0

                tmp_hp = HeatPump(
                    id=idx,
                    pos=pos_hp,
                    orientation=0,
                    inputs=tmp_input,
                    dist_corner_hp=distance_hp_corner,
                    label=tmp_label,
                )
                tmp_hp.recalc_sdf(self.info)
                hp_boxes.append(tmp_hp)
                logging.info(
                    f"HP BOX at {pos_hp} is with ({corner_ll}, {corner_ur}) in domain"
                )
            except:
                logging.warning(f"BOX of HP {idx} at {pos_hp} is not in domain")
        return hp_boxes

    def add_hp(self, hp: "HeatPump", prediction_field: np.ndarray):
        # compose learned fields into large domain with list of ids, pos, orientations
        for i in range(prediction_field.shape[0]):
            for j in range(prediction_field.shape[1]):
                x, y = self.coord_trafo(
                    hp.pos,
                    (i - hp.dist_corner_hp[0], j - hp.dist_corner_hp[1]),
                    hp.orientation,
                )
                if (
                    0 <= x < self.prediction.shape[0]
                    and 0 <= y < self.prediction.shape[1]
                ):
                    self.prediction[x, y] = self.stitching(
                        self.prediction[x, y], prediction_field[i, j]
                    )

    def coord_trafo(self, fixpoint: tuple, position: tuple, orientation: float):
        """
        transform coordinates from domain to hp
        """
        x = (
            fixpoint[0]
            + int(position[0] * cos(orientation))
            + int(position[1] * sin(orientation))
        )
        y = (
            fixpoint[1]
            + int(position[0] * sin(orientation))
            + int(position[1] * cos(orientation))
        )
        return x, y

    def plot(self, fields: str = "t"):
        properties = expand_property_names(fields)
        n_subplots = len(properties)
        if "t" in fields:
            n_subplots += 2
        plt.subplots(n_subplots, 1, sharex=True, figsize=(20, 3 * (n_subplots)))
        idx = 1
        for property in properties:
            plt.subplot(n_subplots, 1, idx)
            if property == "Temperature [C]":
                plt.imshow(self.prediction.T)
                plt.gca().invert_yaxis()
                plt.xlabel("x [cells]")
                plt.ylabel("y [cells]")
                _aligned_colorbar(label=f"Predicted {property}")
                idx += 1
                plt.subplot(n_subplots, 1, idx)
                if self.normed:
                    self.label = self.reverse_norm(self.label, property)
                    self.normed = False
                plt.imshow(abs(self.prediction.T - np.squeeze(self.label.T)))
                plt.gca().invert_yaxis()
                plt.xlabel("x [cells]")
                plt.ylabel("y [cells]")
                _aligned_colorbar(label=f"Absolute error in {property}")
                idx += 1
                plt.subplot(n_subplots, 1, idx)
                plt.imshow(self.label.T)
            elif property == "Original Temperature [C]":
                field = self.prediction_2HP
                property = "1st Prediction of Temperature [C]"
                plt.imshow(field.T)
            else:
                field = self.get_input_field_from_name(property)
                field = self.reverse_norm(field, property)
                plt.imshow(field.T)
            plt.gca().invert_yaxis()
            plt.xlabel("x [cells]")
            plt.ylabel("y [cells]")
            _aligned_colorbar(label=property)
            idx += 1
        plt.savefig("test.png")


class HeatPump:
    def __init__(self, id, pos, orientation, inputs, dist_corner_hp=None, label=None):
        self.id: str = id  # RUN_{ID}
        self.pos: list = np.array(
            [int(pos[0]), int(pos[1])]
        )  # (x,y), cell-ids # TODO(y,x)??
        self.orientation: float = float(orientation)
        self.dist_corner_hp: np.ndarray = (
            dist_corner_hp  # distance from corner of heat pump to corner of box
        )
        self.inputs: np.ndarray = inputs  # extracted from large domain
        self.prediction_1HP = (
            None  # np.ndarray, temperature field, calculated by 1HP-NN
        )
        self.prediction_2HP = (
            None  # np.ndarray, temperature field, calculated by 2HP-NN
        )
        self.interim_outputs: np.ndarray = (
            np.ones(self.inputs[0].shape, dtype="float32") * 10.6
        )  # input for 2HP-NN, size of one input
        self.label = label
        assert (
            self.pos[0] >= 0 and self.pos[1] >= 0
        ), f"Heat pump position at {self.pos} is outside of domain"

    def recalc_sdf(self, info):
        # recalculate sdf per box (cant be done in prepare_dataset because of several hps in one domain)
        # TODO sizedependent... - works as long as boxes have same size in training as in prediction
        index_id = info["Inputs"]["Material ID"]["index"]
        index_sdf = info["Inputs"]["SDF"]["index"]
        loc_hp = self.dist_corner_hp
        assert self.inputs[index_id, loc_hp[0], loc_hp[1]] == 1, f"No HP at {self.pos}"
        self.inputs[index_sdf] = SignedDistanceTransform().sdf(
            self.inputs[index_id].copy(), Tensor(loc_hp)
        )
        assert (
            self.inputs[index_sdf].max() == 1 and self.inputs[index_sdf].min() == 0
        ), "SDF not in [0,1]"

    def apply_nn(self, model):
        input = unsqueeze(Tensor(self.inputs), 0)
        model.eval()
        output = model(input)
        output = output.squeeze().detach().numpy()
        return output

    def get_inputs_for_2hp_nn(self, single_hps):
        hp: HeatPump
        for hp in single_hps:
            # get other hps
            if hp.id != self.id:
                assert (
                    self.interim_outputs.shape == hp.prediction_1HP.shape
                ), "Shapes don't fit - line 241"
                # get overlapping piece of 2nd hp T-box
                rel_pos = hp.pos - self.pos
                zeros2 = [0, 0]
                offset = np.max([-rel_pos, zeros2], axis=0)
                end = hp.prediction_1HP.shape - np.max([rel_pos, zeros2], axis=0)
                tmp_2nd_hp = hp.prediction_1HP[offset[0] : end[0], offset[1] : end[1]]
                # insert at overlapping position in current hp
                offset2 = np.max([rel_pos, zeros2], axis=0)
                end2 = self.interim_outputs.shape - np.max([-rel_pos, zeros2], axis=0)
                self.interim_outputs[
                    offset2[0] : end2[0], offset2[1] : end2[1]
                ] = np.max(
                    [
                        self.interim_outputs[
                            offset2[0] : end2[0], offset2[1] : end2[1]
                        ],
                        tmp_2nd_hp,
                    ],
                    axis=0,
                )

    def save(
        self,
        run_id: str = "",
        dir: str = "HP-Boxes",
        additional_inputs: np.ndarray = None,
        inputs_all: np.ndarray = None,
    ):
        if not os.path.exists(dir):
            os.makedirs(f"{dir}/Inputs")
            os.makedirs(f"{dir}/Labels")
        if (inputs_all != None).any():
            inputs = inputs_all
        elif (additional_inputs != None).any():
            inputs = np.append(self.inputs, additional_inputs, axis=0)
        else:
            inputs = self.inputs
        save(inputs, f"{dir}/Inputs/{run_id}HP_{self.id}.pt")
        save(self.label, f"{dir}/Labels/{run_id}HP_{self.id}.pt")

    def plot_fields(self, n_subplots: int, domain: Domain):
        plt.subplots(n_subplots, 1, sharex=True, figsize=(20, 3 * (n_subplots)))
        idx = 1
        for input_idx, input in enumerate(self.inputs):
            plt.subplot(n_subplots, 1, idx)
            plt.imshow(input.T)
            plt.gca().invert_yaxis()
            plt.xlabel("y [cells]")
            plt.ylabel("x [cells]")
            _aligned_colorbar(label=domain.get_name_from_index(input_idx))
            idx += 1

    def plot_prediction_1HP(self, n_subplots, idx):
        plt.subplot(n_subplots, 1, idx)
        plt.imshow(self.prediction_1HP.T)
        plt.gca().invert_yaxis()
        plt.xlabel("y [cells]")
        plt.ylabel("x [cells]")
        _aligned_colorbar(label="Temperature [C]")

    def plot_prediction_2HP(self, n_subplots, idx):
        plt.subplot(n_subplots, 1, idx)
        plt.imshow(self.prediction_2HP.T)
        plt.gca().invert_yaxis()
        plt.xlabel("y [cells]")
        plt.ylabel("x [cells]")
        _aligned_colorbar(label="Temperature [C]")

    def plot_1HP(self, domain: Domain, dir: str = "HP-Boxes"):
        n_subplots = len(self.inputs) + 1
        self.plot_fields(n_subplots, domain)
        self.plot_prediction_1HP(n_subplots, idx=n_subplots)
        logging.info(f"Saving plot to {dir}/hp_{self.id}.png")
        plt.savefig(f"{dir}/hp_{self.id}.png")

    def plot_2HP(self, domain: Domain, dir: str = "HP-Boxes_2HP"):
        n_subplots = len(self.inputs) + 2
        self.plot_fields(n_subplots, domain)
        self.plot_prediction_1HP(n_subplots, idx=n_subplots - 1)
        self.plot_prediction_2HP(n_subplots, idx=n_subplots)
        plt.savefig(f"{dir}/hp_{self.id}.png")

    def plot(
        self,
        dir: str = "HP-Boxes",
        data_to_plot: np.ndarray = None,
        names: np.ndarray = None,
    ):
        if data_to_plot.any() != None:
            n_subplots = len(data_to_plot)
            assert len(data_to_plot) == len(
                names
            ), "Number of data to plot does not match number of labels"
            plt.subplots(n_subplots, 1, sharex=True, figsize=(20, 3 * (n_subplots)))
            for idx in range(len(data_to_plot)):
                plt.subplot(n_subplots, 1, idx + 1)
                plt.imshow(data_to_plot[idx].T)
                plt.gca().invert_yaxis()
                plt.xlabel("y [cells]")
                plt.ylabel("x [cells]")
                _aligned_colorbar(label=names[idx])
            plt.savefig(f"{dir}/hp_{self.id}.png")
            logging.warning(f"Saving plot to {dir}/hp_{self.id}.png")
        else:
            logging.warning("No data to plot given")


class Stitching:
    def __init__(self, method, background_temperature):
        self.method: str = method
        self.background_temperature: float = background_temperature

    def __call__(self, current_value: float, additional_value: float):
        if self.method == "max":
            return max(current_value, additional_value)
        elif self.method == "add":
            if current_value == self.background_temperature:
                return additional_value
            else:
                return current_value + additional_value - self.background_temperature


def pipeline_prepare_separate_inputs_for_2HPNN(
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
        _,
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

    # prepare 2HP dataset
    list_runs = os.listdir(os.path.join(dataset_domain_path, "Inputs"))
    for run_file in tqdm(list_runs, desc="2HP prepare", total=len(list_runs)):
        run_id = f'{run_file.split(".")[0]}_'
        # print(f"Starting with {run_id}")
        domain = Domain(dataset_domain_path, stitching_method="max", file_name=run_file)
        # generate 1hp-boxes and extract information like perm and ids etc.
        single_hps = domain.extract_hp_boxes()
        # apply learned NN to predict the heat plumes
        hp: HeatPump
        for hp in single_hps:
            hp.prediction_1HP = hp.apply_nn(model_1HP)
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

    # save infos of info file about separated (only 2!) inputs
    save_config_of_separate_inputs(
        domain.info, path=destination_2hp_prep, name_file="info"
    )
    # save command line arguments
    cla = {
        "dataset_large_name": dataset_large_name,
        "model_name_1HP": model_name_1HP,
        "dataset_trained_model_name": dataset_trained_model_name,
        "input_pg": input_pg,
        "model_name_2HP": model_name_2HP,
    }
    save_yaml(cla, path=destination_2hp_prep, name_file="command_line_args")


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


def get_box_corners(
    pos_hp, size_hp_box, distance_hp_corner, domain_shape, run_name: str = "unknown"
):
    corner_ll = pos_hp - np.array(distance_hp_corner)  # corner lower left
    corner_ur = (
        pos_hp + np.array(size_hp_box) - np.array(distance_hp_corner)
    )  # corner upper right

    # if corner_ll[0] < 0 or corner_ur[0] >= domain_shape[0] or corner_ll[1] < 0 or corner_ur[1] >= domain_shape[1]:
    #     # move file from "Inputs" to "broken/Inputs"
    #     logging.warning(f"HP BOX at {pos_hp} is with x=({corner_ll[0]}, {corner_ur[0]}) in x-direction (0, {domain_shape[0]}) or y=({corner_ll[1]}, {corner_ur[1]}) in y-direction (0, {domain_shape[1]}) not in domain for {run_name}")
    #     origin_2hp_prep = "/home/pelzerja/Development/datasets_prepared/2hps_demonstrator/dataset_2hps_1fixed_1000dp_grad_p"
    #     # TODO rm absolute path
    #     shutil.move(os.path.join(origin_2hp_prep, "Inputs", run_name), os.path.join(origin_2hp_prep, "broken", "Inputs", f"{run_name.split('.')[0]}_hp_pos_outside_domain.pt"))
    #     shutil.move(os.path.join(origin_2hp_prep, "Labels", run_name), os.path.join(origin_2hp_prep, "broken", "Labels", run_name))
    assert (
        corner_ll[0] >= 0 and corner_ur[0] < domain_shape[0]
    ), f"HP BOX at {pos_hp} is with x=({corner_ll[0]}, {corner_ur[0]}) in x-direction (0, {domain_shape[0]}) not in domain for {run_name}"
    assert (
        corner_ll[1] >= 0 and corner_ur[1] < domain_shape[1]
    ), f"HP BOX at {pos_hp} is with y=({corner_ll[1]}, {corner_ur[1]}) in y-direction (0, {domain_shape[1]}) not in domain for {run_name}"

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

    args = parser.parse_args()
    args.device = "cpu"

    pipeline_prepare_separate_inputs_for_2HPNN(
        dataset_large_name=args.dataset_large,
        model_name_1HP=args.model,
        dataset_trained_model_name=args.dataset_boxes,
        input_pg=args.input_pg,
        model_name_2HP=args.model_2hp,
        device=args.device,
    )

    beep()
