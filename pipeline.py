import os
import sys
import argparse
import pathlib
import numpy as np
from math import cos, sin
import matplotlib.pyplot as plt
from torch import unsqueeze, load, Tensor
from utils_pipeline import load_info

sys.path.append("/home/pelzerja/pelzerja/test_nn/1HP_NN") # relevant for remote
sys.path.append("/home/pelzerja/Development/1HP_NN")      # relevant for local   
from networks.models import load_model
from utils.visualize_data import _aligned_colorbar
from prepare_dataset import prepare_dataset, expand_property_names


class SimulationDatasetInfo:
    def __init__(self, dataset_path:str):
        self.temp_info = load_info(pathlib.Path(dataset_path))["Labels"]["Temperature [C]"] # expects to have ONE relevant output field, called Temp..
        self.inputs_info = load_info(pathlib.Path(dataset_path))["Inputs"]

    def reverse_norm(self, data:np.ndarray, property:str = "Temperature [C]"):
        if property == "Temperature [C]":
            norm = self.temp_info["norm"]
            max = self.temp_info["max"]
            min = self.temp_info["min"]
            mean = self.temp_info["mean"]
            std = self.temp_info["std"]
        else:
            norm = self.inputs_info[property]["norm"]
            max = self.inputs_info[property]["max"]
            min = self.inputs_info[property]["min"]
            mean = self.inputs_info[property]["mean"]
            std = self.inputs_info[property]["std"]

        if norm=="Rescale":
            out_min, out_max = (0,1)        # Achtung! Hardcoded, values same as in transforms.NormalizeTransform.out_min/max
            delta = max - min
            data = (data - out_min) / (out_max - out_min) * delta + min
        elif norm=="Standardize":
            data = data * std + mean
        elif norm is None:
            pass
        else:
            raise ValueError(f"Normalization type '{self.norm['Norm']}' not recognized")
        return data

class SingleHeatPump:
    def __init__(self, id, pos, orientation, inputs=None, dist_corner_hp=None):
        self.id:str = id                                        # RUN_{ID}
        self.pos:list = np.array([int(pos[0]), int(pos[1])])    #(x,y), cell-ids
        self.orientation:float = float(orientation)
        self.dist_corner_hp: np.ndarray = dist_corner_hp        # distance from corner of heat pump to corner of box
        self.inputs:np.ndarray = inputs                         # extracted from large domain
        self.field = None                                       # np.ndarray, temperature field, calculated by NN
        assert self.pos[0] >= 0 and self.pos[1] >= 0, f"Heat pump position at {self.pos} is outside of domain"


    def apply_nn(self, model, info:SimulationDatasetInfo):
        input = unsqueeze(Tensor(self.inputs), 0)
        model.eval()
        output = model(input)
        output = output.squeeze().detach().numpy()
        output = info.reverse_norm(output, property="Temperature [C]")
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
    def __init__(self, info_path:str, stitching_method:str="max"):
        self.info = load_info(pathlib.Path(info_path))
        self.size:tuple[int, int]           = [self.info["CellsNumber"][1], self.info["CellsNumber"][0]]        # (x, y), cell-ids
        self.background_temperature: float  = 10.6
        self.inputs: np.ndarray             = self.load_datapoint(info_path, case = "Inputs")
        self.label: np.ndarray              = self.load_datapoint(info_path, case = "Labels")
        self.t_field: np.ndarray            = np.ones(self.size) * self.background_temperature
        self.stitching:Stitching            = Stitching(stitching_method, self.background_temperature)

    def load_datapoint(self, dataset_domain_path:str, case:str = "Inputs"):
        # load dataset of large domain
        file_name = "RUN_0.pt"
        file_path = os.path.join(dataset_domain_path, case, file_name)
        data = load(file_path).detach().numpy()
        return data
    
    def get_input_field_from_name(self, name:str):
        field_info = self.info["Inputs"][name]
        field = self.inputs[field_info["index"],:,:]
        return field

    def extract_hp_boxes(self):
        # TODO decide: get hp_boxes based on grad_p or based on v or get squared boxes around hp
        material_ids = self.get_input_field_from_name("Material ID")
        # plt.imshow(np.flip(material_ids.T,axis=0))
        # plt.show()
        size_hp_box = [64, 16] #[128,16]
        distance_hp_corner = [24, 10]
        hp_boxes = []
        pos_hps = np.array(np.where(material_ids == np.max(material_ids))).T
        for idx in range(len(pos_hps)):
            pos_hp = pos_hps[idx]
            corner_ld = pos_hp - np.array(distance_hp_corner)                           # corner left down
            corner_ru = pos_hp + np.array(size_hp_box) - np.array(distance_hp_corner)   # corner right up
            assert corner_ld[0] >= 0 and corner_ru[0] < self.inputs.shape[1], f"HP BOX at {pos_hp} is in x-direction not in domain"
            assert corner_ld[1] >= 0 and corner_ru[1] < self.inputs.shape[2], f"HP BOX at {pos_hp} is in y-direction not in domain"
            # tmp = self.inputs[:, pos_hp[0]-size_hp_box[0]//2:pos_hp[0]+size_hp_box[0]//2, pos_hp[1]-size_hp_box[1]//2:pos_hp[1]+size_hp_box[1]//2]
            tmp_input = self.inputs[:, corner_ld[0]:corner_ru[0], corner_ld[1]:corner_ru[1]]
            tmp_hp = SingleHeatPump(id = f"RUN_{idx}", pos = pos_hp, orientation = 0, inputs = tmp_input, dist_corner_hp=distance_hp_corner)
            hp_boxes.append(tmp_hp)
        return hp_boxes
                
    def add_hp(self, hp:SingleHeatPump):
        for i in range(hp.field.shape[0]):
            for j in range(hp.field.shape[1]):
                x,y = self.coord_trafo(hp.pos, (i-hp.dist_corner_hp[0],j-hp.dist_corner_hp[1]), hp.orientation)
                if 0 <= x < self.t_field.shape[0] and 0 <= y < self.t_field.shape[1]:
                    self.t_field[x,y] = self.stitching(self.t_field[x, y], hp.field[i, j])

    def coord_trafo(self, fixpoint:tuple, position:tuple, orientation:float):
        """
        transform coordinates from domain to hp
        """
        x = fixpoint[0] +int(position[0]*cos(orientation))+int(position[1]*sin(orientation))
        y = fixpoint[1] +int(position[0]*sin(orientation))+int(position[1]*cos(orientation))
        return x, y

    def plot_field(self, fields:str = "t", info:SimulationDatasetInfo = None):
        properties = expand_property_names(fields)
        n_subplots = len(properties)
        if "t" in fields:
            n_subplots += 1
        _, axes = plt.subplots(n_subplots, 1, sharex=True,figsize=(20, 3*(n_subplots)))
        idx = 1
        for property in properties:
            plt.subplot(n_subplots, 1, idx)

            if property == "Temperature [C]":
                plt.imshow(self.t_field.T)
                plt.gca().invert_yaxis()
                plt.xlabel("y [cells]")
                plt.ylabel("x [cells]")
                _aligned_colorbar(label=f"Predicted {property}")
                idx+=1
                plt.subplot(n_subplots, 1, idx)
                self.label = info.reverse_norm(self.label, property)
                plt.imshow(self.label.T)
            else:
                field = self.get_input_field_from_name(property)
                field = info.reverse_norm(field, property)
                plt.imshow(field.T)
            plt.gca().invert_yaxis()
            plt.xlabel("y [cells]")
            plt.ylabel("x [cells]")
            # plt.suptitle(f"{property}")
            _aligned_colorbar(label=property)
            idx += 1
        plt.show()

def pipeline(dataset_large_name:str, model_name:str, dataset_trained_model_name:str, device:str="cuda:0"):
    """
    assumptions:
    - 1hp-boxes are generated already
    - network is trained
    """
    remote = False

    # prepare large dataset if not done yet
    if os.path.exists("/scratch/sgs/pelzerja/"):
        remote = True

    if remote:
        raw_data_dir = "/scratch/sgs/pelzerja/datasets/2hps_demonstrator"
        datasets_path = "/home/pelzerja/pelzerja/test_nn/datasets_prepared/2hps_demonstrator"
        models_dir = "/home/pelzerja/pelzerja/test_nn/1HP_NN/runs"
        model_dataset_dir = "/home/pelzerja/pelzerja/test_nn/datasets_prepared"
    else:
        raw_data_dir = "/home/pelzerja/Development/datasets/2hps_demonstrator"
        datasets_path = "/home/pelzerja/Development/datasets_prepared/2hps_demonstrator"
        models_dir = "/home/pelzerja/Development/1HP_NN/runs"
        model_dataset_dir = "/home/pelzerja/Development/datasets_prepared/1HP_NN"
    dataset_domain_path = os.path.join(datasets_path, dataset_large_name)
    datasets_path_model_trained_with = os.path.join(model_dataset_dir, dataset_trained_model_name)
    model_path = os.path.join(models_dir, model_name)
    info_dataset_model_trained_with = load_info(pathlib.Path(datasets_path_model_trained_with))

    if not os.path.exists(dataset_domain_path):
        prepare_dataset(raw_data_directory = raw_data_dir,
                        datasets_path = datasets_path,
                        dataset_name = dataset_large_name,
                        input_variables = "pksi",
                        power2trafo = False,
                        info = info_dataset_model_trained_with) # norm with data from dataset that NN was trained with!
    domain = ComposedDomain(dataset_domain_path, stitching_method="max")
    # generate 1hp-boxes and extract information like perm and ids+pos+orientations-list, BEFORE: choose 1hp-boxes
    # TODO v-orientation currently not possible -> p-orientation for now? or kick out orientation and work with square boxes?
    single_hps = domain.extract_hp_boxes()

    # apply learned NN to predict the heat plumes
    model = load_model({"model_choice": "unet", "in_channels": 4}, model_path, "model", device)
    dataset_boxes_info = SimulationDatasetInfo(datasets_path_model_trained_with)
    # single_hps = get_single_hps(dataset_boxes_info)
    for hp in single_hps:
        hp.apply_nn(model, dataset_boxes_info)
        # compose learned fields into large domain with list of ids, pos, orientations
        domain.add_hp(hp)
    domain.plot_field("tkis", dataset_boxes_info)
    #LATER: smooth large domain and extend heat plumes

# def get_single_hps(dataset_info:str):
#     # Zuordnung: RUN_ID.pt file <-> id+pos+orientation
#     csv_path = os.path.join(dataset_info.path, "hp_positions.csv")
#     single_hps = []
#     with open(csv_path, newline='') as csvfile:
#         reader = csv.reader(csvfile, delimiter=',')
#         next(reader)
#         for row in reader:
#             single_hps.append(SingleHeatPump(id=row[0], pos=[row[1], row[2]], orientation=row[3]))
#     return single_hps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_large", type=str, default="large_2hps_simulation")
    parser.add_argument("--model", type=str, default="current_unet_benchmark_dataset_2d_100datapoints_assumedsteadystate")
    parser.add_argument("--dataset_boxes", type=str, default="benchmark_dataset_2d_100datapoints_assumedsteadystate")
    args = parser.parse_args()
    args.device = "cpu"
    pipeline(dataset_large_name=args.dataset_large, model_name=args.model, dataset_trained_model_name=args.dataset_boxes, device=args.device)