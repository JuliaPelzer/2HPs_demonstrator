import os
import sys
import argparse
import pathlib
import numpy as np
from math import cos, sin
import matplotlib.pyplot as plt
from torch import squeeze, unsqueeze, load, Tensor, save

sys.path.append("/home/pelzerja/pelzerja/test_nn/1HP_NN") # relevant for remote
sys.path.append("/home/pelzerja/Development/1HP_NN")      # relevant for local   
from networks.models import load_model
from utils.visualize_data import _aligned_colorbar
from prepare_dataset import prepare_dataset, expand_property_names
from data.utils import load_yaml
from data.transforms import SignedDistanceTransform

class Domain:
    def __init__(self, info_path:str, stitching_method:str="max"):
        self.info = load_yaml(info_path, "info")
        self.size:tuple[int, int]           = [self.info["CellsNumber"][0], self.info["CellsNumber"][1]]        # (x, y), cell-ids
        self.background_temperature: float  = 10.6
        self.inputs: np.ndarray             = self.load_datapoint(info_path, case = "Inputs")
        self.label: np.ndarray              = self.load_datapoint(info_path, case = "Labels")
        self.t_field: np.ndarray            = np.ones(self.size) * self.background_temperature
        self.stitching:Stitching            = Stitching(stitching_method, self.background_temperature)
        try:
            print(f"Pressure: {self.get_input_field_from_name('Liquid Pressure [Pa]').max(), self.get_input_field_from_name('Liquid Pressure [Pa]').min()}")
        except:
            print(f"Pressure gradient: {self.get_input_field_from_name('Pressure Gradient [-]').max(), self.get_input_field_from_name('Pressure Gradient [-]').min()}")
            assert self.get_input_field_from_name('Pressure Gradient [-]').max() <= 1 and self.get_input_field_from_name('Pressure Gradient [-]').min() >= 0, "Pressure Gradient [-] not in range (0,1)"
        print(f"Permeability: {self.get_input_field_from_name('Permeability X [m^2]').max(), self.get_input_field_from_name('Permeability X [m^2]').min()}")
        assert self.get_input_field_from_name('Permeability X [m^2]').max() <= 1 and self.get_input_field_from_name('Permeability X [m^2]').min() >= 0, "Permeability X [m^2] not in range (0,1)"
        
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

    def reverse_norm(self, data:np.ndarray, property:str = "Temperature [C]"):
        try:
            norm = self.info["Inputs"][property]["norm"]
            max =  self.info["Inputs"][property]["max"]
            min =  self.info["Inputs"][property]["min"]
            mean = self.info["Inputs"][property]["mean"]
            std =  self.info["Inputs"][property]["std"]
        except:
            norm = self.info["Labels"][property]["norm"]
            max =  self.info["Labels"][property]["max"]
            min =  self.info["Labels"][property]["min"]
            mean = self.info["Labels"][property]["mean"]
            std =  self.info["Labels"][property]["std"]

        if norm=="Rescale":
            out_min, out_max = (0,1)        # TODO Achtung! Hardcoded, values same as in transforms.NormalizeTransform.out_min/max
            delta = max - min
            data = (data - out_min) / (out_max - out_min) * delta + min
        elif norm=="Standardize":
            data = data * std + mean
        elif norm is None:
            pass
        else:
            raise ValueError(f"Normalization type '{self.norm['Norm']}' not recognized")
        return data

    def extract_hp_boxes(self):
        # TODO decide: get hp_boxes based on grad_p or based on v or get squared boxes around hp
        material_ids = self.get_input_field_from_name("Material ID")
        size_hp_box = [self.info["CellsNumberPrior"][0], self.info["CellsNumberPrior"][1]]
        distance_hp_corner = [self.info["PositionHPPrior"][0], self.info["PositionHPPrior"][1]]
        hp_boxes = []
        pos_hps = np.array(np.where(material_ids == np.max(material_ids))).T
        for idx in range(len(pos_hps)):
            pos_hp = pos_hps[idx]
            corner_ll = pos_hp - np.array(distance_hp_corner)                           # corner lower left
            corner_ur = pos_hp + np.array(size_hp_box) - np.array(distance_hp_corner)   # corner upper right
            assert corner_ll[0] >= 0 and corner_ur[0] < self.inputs.shape[1], f"HP BOX at {pos_hp} is with ({corner_ll[0]}, {corner_ur[0]}) in x-direction (0, {self.inputs.shape[1]}) not in domain"
            assert corner_ll[1] >= 0 and corner_ur[1] < self.inputs.shape[2], f"HP BOX at {pos_hp} is with ({corner_ll[1]}, {corner_ur[1]}) in y-direction (0, {self.inputs.shape[2]}) not in domain"
            tmp_input = self.inputs[:, corner_ll[0]:corner_ur[0], corner_ll[1]:corner_ur[1]].copy()
            tmp_label = self.label[:, corner_ll[0]:corner_ur[0], corner_ll[1]:corner_ur[1]].copy()

            tmp_mat_ids = np.array(np.where(tmp_input == np.max(material_ids))).T
            if len(tmp_mat_ids) > 1:
                for i in range(len(tmp_mat_ids)):
                    tmp_pos = tmp_mat_ids[i]
                    if (tmp_pos[1:2] != distance_hp_corner).all():
                        tmp_input[tmp_pos[0],tmp_pos[1], tmp_pos[2]] = 0
            tmp_hp = HeatPump(id = f"RUN_{idx}", pos = pos_hp, orientation = 0, inputs = tmp_input, dist_corner_hp=distance_hp_corner, tmp_label = tmp_label)
            tmp_hp.recalc_sdf(self.info)
            hp_boxes.append(tmp_hp)
        return hp_boxes
                
    def add_hp(self, hp):
        # compose learned fields into large domain with list of ids, pos, orientations
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
    
    def reverse_coord_trafo():
        # TODO
        pass

    def plot_field(self, fields:str = "t"):
        properties = expand_property_names(fields)
        n_subplots = len(properties)
        if "t" in fields:
            n_subplots += 2
        plt.subplots(n_subplots, 1, sharex=True,figsize=(20, 3*(n_subplots)))
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
                self.label = self.reverse_norm(self.label, property)
                plt.imshow(abs(self.t_field.T-np.squeeze(self.label.T)))
                plt.gca().invert_yaxis()
                plt.xlabel("y [cells]")
                plt.ylabel("x [cells]")
                _aligned_colorbar(label=f"Absolute error in {property}")
                idx+=1
                plt.subplot(n_subplots, 1, idx)
                plt.imshow(self.label.T)
            else:
                field = self.get_input_field_from_name(property)
                field = self.reverse_norm(field, property)
                plt.imshow(field.T)
            plt.gca().invert_yaxis()
            plt.xlabel("y [cells]")
            plt.ylabel("x [cells]")
            _aligned_colorbar(label=property)
            idx += 1
        plt.savefig("test.png")

class HeatPump:
    def __init__(self, id, pos, orientation, inputs=None, dist_corner_hp=None, tmp_label=None):
        self.id:str = id                                        # RUN_{ID}
        self.pos:list = np.array([int(pos[0]), int(pos[1])])    #(x,y), cell-ids
        self.orientation:float = float(orientation)
        self.dist_corner_hp: np.ndarray = dist_corner_hp        # distance from corner of heat pump to corner of box
        self.inputs:np.ndarray = inputs                         # extracted from large domain
        self.field = None                                       # np.ndarray, temperature field, calculated by NN
        self.tmp_label = tmp_label
        assert self.pos[0] >= 0 and self.pos[1] >= 0, f"Heat pump position at {self.pos} is outside of domain"

    def recalc_sdf(self, info):
        # recalculate sdf per box (cant be done in prepare_dataset because of several hps in one domain)
        # TODO sizedependent... - works as long as boxes have same size in training as in prediction
        index_id = info["Inputs"]["Material ID"]["index"]
        index_sdf = info["Inputs"]["SDF"]["index"]
        loc_hp = self.dist_corner_hp
        assert self.inputs[index_id, loc_hp[0], loc_hp[1]] == 1, f"No HP at {self.pos}"
        self.inputs[index_sdf] = SignedDistanceTransform().sdf(self.inputs[index_id].copy(), Tensor(loc_hp))

    def apply_nn(self, model, domain:Domain):
        input = unsqueeze(Tensor(self.inputs), 0)
        model.eval()
        output = model(input)
        output = output.squeeze().detach().numpy()
        output = domain.reverse_norm(output, property="Temperature [C]")
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

def pipeline(dataset_large_name:str, model_name:str, dataset_trained_model_name:str, device:str="cuda:0"):
    """
    assumptions:
    - 1hp-boxes are generated already
    - network is trained
    """
    # prepare large dataset if not done yet
    datasets_raw_domain_dir, datasets_prepared_domain_dir, dataset_domain_path, datasets_model_trained_with_path, model_path = set_paths(dataset_large_name, model_name, dataset_trained_model_name)
    if not os.path.exists(dataset_domain_path):
        prepare_dataset(raw_data_directory = datasets_raw_domain_dir,
                        datasets_path = datasets_prepared_domain_dir,
                        dataset_name = dataset_large_name,
                        input_variables = "pksi",
                        power2trafo = False,
                        info = load_yaml(datasets_model_trained_with_path, "info")) # norm with data from dataset that NN was trained with!
    else:
        print(f"Domain {dataset_domain_path} already prepared")
        
    domain = Domain(dataset_domain_path, stitching_method="max")
    # generate 1hp-boxes and extract information like perm and ids+pos+orientations-list, BEFORE: choose 1hp-boxes
    # TODO v-orientation currently not possible -> p-orientation for now? or kick out orientation and work with square boxes?
    single_hps = domain.extract_hp_boxes()

    # apply learned NN to predict the heat plumes
    model = load_model({"model_choice": "unet", "in_channels": 4}, model_path, "model", device)
    for hp in single_hps:
        hp.apply_nn(model, domain)
        domain.add_hp(hp)
    domain.plot_field("tpki")
    #TODO LATER: smooth large domain and extend heat plumes

def set_paths(dataset_large_name:str, model_name:str, dataset_trained_model_name:str):
    remote = False
    if os.path.exists("/scratch/sgs/pelzerja/"):
        remote = True
    if remote:
        datasets_raw_domain_dir = "/scratch/sgs/pelzerja/datasets/2hps_demonstrator"
        datasets_prepared_domain_dir = "/home/pelzerja/pelzerja/test_nn/datasets_prepared/2hps_demonstrator"
        models_dir = "/home/pelzerja/pelzerja/test_nn/1HP_NN/runs"
        datasets_prepared_1hp_dir = "/home/pelzerja/pelzerja/test_nn/datasets_prepared"
    else:
        datasets_raw_domain_dir = "/home/pelzerja/Development/datasets/2hps_demonstrator"
        datasets_prepared_domain_dir = "/home/pelzerja/Development/datasets_prepared/2hps_demonstrator"
        models_dir = "/home/pelzerja/Development/1HP_NN/runs"
        datasets_prepared_1hp_dir = "/home/pelzerja/Development/datasets_prepared/1HP_NN"
    dataset_domain_path = os.path.join(datasets_prepared_domain_dir, dataset_large_name)
    datasets_model_trained_with_path = os.path.join(datasets_prepared_1hp_dir, dataset_trained_model_name)
    model_path = os.path.join(models_dir, model_name)

    return datasets_raw_domain_dir, datasets_prepared_domain_dir, dataset_domain_path, datasets_model_trained_with_path, model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_large", type=str, default="benchmark_dataset_2d_2hps_iso_perm")
    parser.add_argument("--model", type=str, default="current_unet_benchmark_dataset_2d_100datapoints")
    parser.add_argument("--dataset_boxes", type=str, default="benchmark_dataset_2d_100datapoints")
    args = parser.parse_args()
    args.device = "cpu"
    pipeline(dataset_large_name=args.dataset_large, model_name=args.model, dataset_trained_model_name=args.dataset_boxes, device=args.device)