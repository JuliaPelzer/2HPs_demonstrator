import logging
import pathlib
import sys
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor, save, unsqueeze

sys.path.append("/home/pelzerja/pelzerja/test_nn/1HP_NN")  # relevant for remote
sys.path.append("/home/pelzerja/Development/1HP_NN")  # relevant for local
from data.transforms import SignedDistanceTransform
from utils.visualize_data import _aligned_colorbar

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
                ), "Shapes don't fit - line 366"
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

    def save(self, run_id: str = "", dir: str = "HP-Boxes", additional_inputs: np.ndarray = None, inputs_all: np.ndarray = None,):
        dir_in = dir / "Inputs"
        dir_in.mkdir(parents=True, exist_ok=True)
        pathlib.Path(dir, "Inputs").mkdir(parents=True, exist_ok=True)
        pathlib.Path(dir, "Labels").mkdir(parents=True, exist_ok=True)
        if (inputs_all != None).any():
            inputs = inputs_all
        elif (additional_inputs != None).any():
            inputs = np.append(self.inputs, additional_inputs, axis=0)
        else:
            inputs = self.inputs
        save(inputs, f"{dir}/Inputs/{run_id}HP_{self.id}.pt")
        save(self.label, f"{dir}/Labels/{run_id}HP_{self.id}.pt")

    def plot_fields(self, n_subplots: int, domain: "Domain"):
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

    def plot_1HP(self, domain: "Domain", dir: str = "HP-Boxes"):
        n_subplots = len(self.inputs) + 1
        self.plot_fields(n_subplots, domain)
        self.plot_prediction_1HP(n_subplots, idx=n_subplots)
        logging.info(f"Saving plot to {dir}/hp_{self.id}.png")
        plt.savefig(f"{dir}/hp_{self.id}.png")

    def plot_2HP(self, domain: "Domain", dir: str = "HP-Boxes_2HP"):
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

