import os
import sys
import csv
import yaml
import pathlib
import numpy as np
from math import cos, sin
import matplotlib.pyplot as plt
try:
    sys.path.append("/home/pelzerja/pelzerja/test_nn/1HP_NN")
except:
    sys.path.append("/home/pelzerja/Development/1HP_NN")

from torch import unsqueeze, load

from networks.models import load_model
from utils.visualize_data import _aligned_colorbar
from prepare_dataset import prepare_dataset

class SimulationDatasetInfo:
    def __init__(self, dataset_path:str):
        self.path = pathlib.Path(dataset_path)
        self.info = self.__load_info()["Labels"]["Temperature [C]"] # expects to have ONE relevant output field, called Temp..
        self.cells = self.__load_info()["CellsNumber"] #list of 3 (x,y,z)[-]
        self.cells_size = self.__load_info()["CellsSize"] #list of 3 (x,y,z)[m]
        self.norm = self.info["norm"]

    def __load_info(self):
        with open(self.path.joinpath("info.yaml"), "r") as f:
            info = yaml.safe_load(f)
        return info

    def reverse_norm(self, data:np.ndarray):
        if self.norm=="Rescale":
            out_min, out_max = (0,1)        # Achtung! Hardcoded, values same as in transforms.NormalizeTransform.out_min/max
            delta = self.info["max"] - self.info["min"]
            data = (data - out_min) / (out_max - out_min) * delta + self.info["min"]
        elif self.norm=="Standardize":
            data = data * self.info["std"] + self.info["mean"]
        elif self.norm is None:
            pass
        else:
            raise ValueError(f"Normalization type '{self.norm['Norm']}' not recognized")
        return data

class SingleHeatPump:
    def __init__(self, id, pos, orientation):
        self.id:str = id                                        # RUN_{ID}
        self.pos:list = np.array([int(pos[0]), int(pos[1])])    #(x,y), cell-ids
        self.orientation:float = float(orientation)
        self.field = None                                       # np.ndarray, temperature field
        assert self.pos[0] >= 0 and self.pos[1] >= 0, f"Heat pump position at {self.pos} is outside of domain"


    def apply_nn(self, model, info:SimulationDatasetInfo):
            input = load(os.path.join(info.path, "Inputs", self.id + ".pt"))
            input = unsqueeze(input, 0)
            model.eval()
            output = model(input)
            output = output.squeeze().detach().numpy()
            output = info.reverse_norm(output)
            self.field = output

class Stitching:
    def __init__(self, method, background_temperature):
        self.method:str = method
        self.background_temperature:float = background_temperature

    def __call__(self, current_value:float, additional_value:float):
        if self.method=="max":
            return max(current_value, additional_value)
        elif self.method=="add":
            if current_value == self.background_temperature:
                return additional_value
            else:
                return current_value + additional_value - self.background_temperature

class ComposedDomain:
    def __init__(self, size, stitching_method:str="max"):
        self.size:tuple[int, int] = size        # (x, y), cell-ids
        self.background_temperature = 10.6
        self.field = np.ones(self.size)*self.background_temperature
        self.stitching = Stitching(stitching_method, self.background_temperature)

    def add_hp(self, hp:SingleHeatPump):
        distance_hp_corner = [int(50/5),int(120/5)] # TODO hardcoded, should be in hp
        box_corner = hp.pos
        for j in range(hp.field.shape[0]):
            for i in range(hp.field.shape[1]):
                x,y = self.coord_trafo(box_corner, (i-distance_hp_corner[0],j-distance_hp_corner[1]), hp.orientation)
                if 0 <= x < self.field.shape[1] and 0 <= y < self.field.shape[0]:
                    self.field[y,x] = self.stitching(self.field[y, x], hp.field[j,i])
                # TODO get rid of periodic BCs

    def coord_trafo(self, fixpoint:tuple, position:tuple, orientation:float):
        """
        transform coordinates from domain to hp
        """
        x = fixpoint[0] +int(position[0]*cos(orientation))+int(position[1]*sin(orientation))
        y = fixpoint[1] +int(position[0]*sin(orientation))+int(position[1]*cos(orientation))
        return x, y

    def plot_field(self):
        # plt.imshow(np.flip(self.field.T, axis=0))
        plt.imshow(self.field.T)
        plt.gca().invert_yaxis()
        plt.xlabel("y [cells]")
        plt.ylabel("x [cells]")
        plt.suptitle("Temperature field")
        _aligned_colorbar()
        plt.show()

def pipeline():
    """
    assumptions:
    - 1hp-boxes are generated already
    - network is trained
    """
    device:str="cuda:0"

    # # prepare large dataset if not done yet
    # large_dataset_name = "large_2hps_simulation"
    # if not os.path.exists(os.path.join("/home/pelzerja/pelzerja/test_nn/datasets_prepared", large_dataset_name)):
    #     prepare_dataset(raw_data_directory = "/scratch/sgs/pelzerja/datasets/2hps_demonstrator/",
    #                     datasets_path = "/home/pelzerja/pelzerja/test_nn/datasets_prepared/2hps_demonstrator",
    #                     dataset_name = large_dataset_name,
    #                     input_variables = "pksi",)

    # LATER generate 1hp-boxes and extract information like perm and ids+pos+orientations-list, NOW: choose 1hp-boxes
    # for cell with mat-id == 1 (i.e. pos of hp):
    #   - get hp-id, hp-pos, v-orientation
    #   - get perm field etc. for NN input
    # ! orientation currently not possible -> square boxes


    # apply learned NN to predict the heat plumes
    model = get_model("current_unet_benchmark_dataset_2d_100datapoints_assumedsteadystate", device)
    name_dataset = "benchmark_dataset_2d_100datapoints_assumedsteadystate"
    dataset_path = os.path.join("/home/pelzerja/pelzerja/test_nn/datasets_prepared", name_dataset)
    dataset_info = SimulationDatasetInfo(dataset_path)
    size_domain = dataset_info.cells
    domain = ComposedDomain(size_domain, stitching_method="max")
    single_hps = get_single_hps(dataset_info)
    for hp in single_hps:
        hp.apply_nn(model, dataset_info)
        domain.add_hp(hp)
    domain.plot_field()

    # compose learned fields into large domain with list of ids, pos, orientations
    # get size of large domain

    # smooth large domain

def get_model(name_model_path, device):
    model_path = os.path.join("/home/pelzerja/pelzerja/test_nn/1HP_NN/runs", name_model_path)
    model = load_model({"model_choice": "unet", "in_channels": 4}, model_path, "model", device)
    return model

def get_single_hps(dataset_info:str):
    # Zuordnung: RUN_ID.pt file <-> id+pos+orientation
    csv_path = os.path.join(dataset_info.path, "hp_positions.csv")
    single_hps = []
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            single_hps.append(SingleHeatPump(id=row[0], pos=[row[1], row[2]], orientation=row[3]))
    return single_hps

if __name__ == "__main__":
    pipeline()