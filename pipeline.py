import os
import sys
import argparse
import pathlib
import logging
from tqdm.auto import tqdm
import numpy as np
from math import cos, sin
import matplotlib.pyplot as plt
from torch import squeeze, unsqueeze, load, Tensor, save

sys.path.append("/home/pelzerja/pelzerja/test_nn/1HP_NN") # relevant for remote
sys.path.append("/home/pelzerja/Development/1HP_NN")      # relevant for local   
from networks.models import load_model
from utils.visualize_data import _aligned_colorbar
from prepare_dataset import prepare_dataset, expand_property_names
from data.utils import load_yaml, save_yaml
from data.transforms import SignedDistanceTransform

class Domain:
    def __init__(self, info_path:str, stitching_method:str="max", file_name:str="RUN_0.pt"):
        self.info = load_yaml(info_path, "info")
        self.size:tuple[int, int]           = [self.info["CellsNumber"][0], self.info["CellsNumber"][1]]        # (x, y), cell-ids
        self.background_temperature: float  = 10.6
        self.inputs: np.ndarray             = self.load_datapoint(info_path, case = "Inputs", file_name=file_name)
        self.label: np.ndarray              = self.load_datapoint(info_path, case = "Labels", file_name=file_name)
        self.prediction: np.ndarray         = np.ones(self.size) * self.background_temperature
        self.prediction_2HP: np.ndarray     = np.ones(self.size) * self.background_temperature
        self.stitching:Stitching            = Stitching(stitching_method, self.background_temperature)
        self.normed:bool                    = True
        assert self.get_input_field_from_name('Permeability X [m^2]').max() <= 1 and self.get_input_field_from_name('Permeability X [m^2]').min() >= 0, "Permeability X [m^2] not in range (0,1)"
        try:
            p_related_name = 'Pressure Gradient [-]'
            p_related_field = self.get_input_field_from_name(p_related_name)
        except:
            p_related_name = 'Liquid Pressure [Pa]'
            p_related_field = self.get_input_field_from_name(p_related_name)
        print(f"{p_related_name} in range ({p_related_field.max()}, {p_related_field.min()})")
        assert p_related_field.max() <= 1 and p_related_field.min() >= 0, f"{p_related_name} not in range (0,1) but {p_related_field.max(), p_related_field.min()}"
        
    def load_datapoint(self, dataset_domain_path:str, case:str = "Inputs", file_name = "RUN_0.pt"):
        # load dataset of large domain
        file_path = os.path.join(dataset_domain_path, case, file_name)
        data = load(file_path).detach().numpy()
        return data
    
    def get_index_from_name(self, name:str):
        return self.info["Inputs"][name]["index"]
    
    def get_name_from_index(self, index:int):
        for property, values in self.info["Inputs"].items():
            if values["index"]==index:
                return property
    
    def get_input_field_from_name(self, name:str):
        field_idx = self.get_index_from_name(name)
        field = self.inputs[field_idx,:,:]
        return field

    def norm(self, data:np.ndarray, property:str = "Temperature [C]"):
        norm_fct, max_val, min_val, mean_val, std_val = self.get_norm_info(property)

        if norm_fct=="Rescale":
            out_min, out_max = (0,1)        # TODO Achtung! Hardcoded, values same as in transforms.NormalizeTransform.out_min/max
            delta = max_val - min_val
            data = (data - min_val) / delta * (out_max - out_min) + out_min
        elif norm_fct=="Standardize":
            data = (data - mean_val) / std_val
        elif norm_fct is None:
            pass
        else:
            raise ValueError(f"Normalization type '{self.norm['Norm']}' not recognized")
        return data
    
    def reverse_norm(self, data:np.ndarray, property:str = "Temperature [C]"):
        norm_fct, max_val, min_val, mean_val, std_val = self.get_norm_info(property)

        if norm_fct=="Rescale":
            out_min, out_max = (0,1)        # TODO Achtung! Hardcoded, values same as in transforms.NormalizeTransform.out_min/max
            delta = max_val - min_val
            data = (data - out_min) / (out_max - out_min) * delta + min_val
        elif norm_fct=="Standardize":
            data = data * std_val + mean_val
        elif norm_fct is None:
            pass
        else:
            raise ValueError(f"Normalization type '{self.norm_fct['Norm']}' not recognized")
        return data
    
    def get_norm_info(self, property:str = "Temperature [C]"):
        try:
            norm_fct = self.info["Inputs"][property]["norm"]
            max_val =  self.info["Inputs"][property]["max"]
            min_val =  self.info["Inputs"][property]["min"]
            mean_val = self.info["Inputs"][property]["mean"]
            std_val =  self.info["Inputs"][property]["std"]
        except:
            norm_fct = self.info["Labels"][property]["norm"]
            max_val =  self.info["Labels"][property]["max"]
            min_val =  self.info["Labels"][property]["min"]
            mean_val = self.info["Labels"][property]["mean"]
            std_val =  self.info["Labels"][property]["std"]
        return norm_fct, max_val, min_val, mean_val, std_val

    def extract_hp_boxes(self):
        # TODO decide: get hp_boxes based on grad_p or based on v or get squared boxes around hp
        material_ids = self.get_input_field_from_name("Material ID")
        size_hp_box = [self.info["CellsNumberPrior"][0], self.info["CellsNumberPrior"][1]]
        distance_hp_corner = [self.info["PositionHPPrior"][0], self.info["PositionHPPrior"][1]]
        hp_boxes = []
        pos_hps = np.array(np.where(material_ids == np.max(material_ids))).T
        for idx in range(len(pos_hps)):
            pos_hp = pos_hps[idx]
            corner_ll, corner_ur = get_box_corners(pos_hp, size_hp_box, distance_hp_corner, self.inputs.shape[1:])
            tmp_input = self.inputs[:, corner_ll[0]:corner_ur[0], corner_ll[1]:corner_ur[1]].copy()
            tmp_label = self.label[:, corner_ll[0]:corner_ur[0], corner_ll[1]:corner_ur[1]].copy()

            tmp_mat_ids = np.array(np.where(tmp_input == np.max(material_ids))).T
            if len(tmp_mat_ids) > 1:
                for i in range(len(tmp_mat_ids)):
                    tmp_pos = tmp_mat_ids[i]
                    if (tmp_pos[1:2] != distance_hp_corner).all():
                        tmp_input[tmp_pos[0],tmp_pos[1], tmp_pos[2]] = 0

            tmp_hp = HeatPump(id = idx, pos = pos_hp, orientation = 0, inputs = tmp_input, dist_corner_hp=distance_hp_corner, label = tmp_label)
            tmp_hp.recalc_sdf(self.info)
            hp_boxes.append(tmp_hp)
            logging.info(f"HP BOX at {pos_hp} is with ({corner_ll}, {corner_ur}) in domain")
        return hp_boxes
                
    def add_hp(self, hp: "HeatPump", prediction_field:np.ndarray):
        # compose learned fields into large domain with list of ids, pos, orientations
        for i in range(prediction_field.shape[0]):
            for j in range(prediction_field.shape[1]):
                x,y = self.coord_trafo(hp.pos, (i-hp.dist_corner_hp[0],j-hp.dist_corner_hp[1]), hp.orientation)
                if 0 <= x < self.prediction.shape[0] and 0 <= y < self.prediction.shape[1]:
                    self.prediction[x,y] = self.stitching(self.prediction[x, y], prediction_field[i, j])
        
    def overwrite_boxes_prediction_1HP(self, hp: "HeatPump"):
        # overwrite hp.prediction_1HP (originally only information from 1HP) with overlapping predicted temperature fields from domain.prediction
        # therefor extract self.prediction in area of hp
        corner_ll, corner_ur = get_box_corners(hp.pos, hp.prediction_1HP.shape, hp.dist_corner_hp, self.prediction.shape)
        field = self.prediction[corner_ll[0]:corner_ur[0], corner_ll[1]:corner_ur[1]].copy()
        hp.prediction_1HP = field

        # overwrite information in info.yaml about "Inputs/Original Temperature [C]" with "Labels/Temperature [C]"
        tmp_index = self.info["Inputs"]["Original Temperature [C]"]["index"]
        self.info["Inputs"]["Original Temperature [C]"] = self.info["Labels"]["Temperature [C]"].copy()
        self.info["Inputs"]["Original Temperature [C]"]["index"] = tmp_index # keep the index!

    def coord_trafo(self, fixpoint:tuple, position:tuple, orientation:float):
        """
        transform coordinates from domain to hp
        """
        x = fixpoint[0] +int(position[0]*cos(orientation))+int(position[1]*sin(orientation))
        y = fixpoint[1] +int(position[0]*sin(orientation))+int(position[1]*cos(orientation))
        return x, y
    
    def plot(self, fields:str = "t"):
        properties = expand_property_names(fields)
        n_subplots = len(properties)
        if "t" in fields:
            n_subplots += 2
        plt.subplots(n_subplots, 1, sharex=True,figsize=(20, 3*(n_subplots)))
        idx = 1
        for property in properties:
            plt.subplot(n_subplots, 1, idx)
            if property == "Temperature [C]":
                plt.imshow(self.prediction.T)
                plt.gca().invert_yaxis()
                plt.xlabel("y [cells]")
                plt.ylabel("x [cells]")
                _aligned_colorbar(label=f"Predicted {property}")
                idx+=1
                plt.subplot(n_subplots, 1, idx)
                if self.normed:
                    self.label = self.reverse_norm(self.label, property)
                    self.normed = False
                plt.imshow(abs(self.prediction.T-np.squeeze(self.label.T)))
                plt.gca().invert_yaxis()
                plt.xlabel("y [cells]")
                plt.ylabel("x [cells]")
                _aligned_colorbar(label=f"Absolute error in {property}")
                idx+=1
                plt.subplot(n_subplots, 1, idx)
                plt.imshow(self.label.T)
            elif property=="Original Temperature [C]":
                field = self.prediction_2HP
                property = "1st Prediction of Temperature [C]"
                plt.imshow(field.T)
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
    def __init__(self, id, pos, orientation, inputs=None, dist_corner_hp=None, label=None):
        self.id:str = id                                        # RUN_{ID}
        self.pos:list = np.array([int(pos[0]), int(pos[1])])    #(x,y), cell-ids
        self.orientation:float = float(orientation)
        self.dist_corner_hp: np.ndarray = dist_corner_hp        # distance from corner of heat pump to corner of box
        self.inputs:np.ndarray = inputs                         # extracted from large domain
        self.prediction_1HP = None                              # np.ndarray, temperature field, calculated by 1HP-NN
        self.prediction_2HP = None                              # np.ndarray, temperature field, calculated by 2HP-NN
        self.label = label
        assert self.pos[0] >= 0 and self.pos[1] >= 0, f"Heat pump position at {self.pos} is outside of domain"

    def recalc_sdf(self, info):
        # recalculate sdf per box (cant be done in prepare_dataset because of several hps in one domain)
        # TODO sizedependent... - works as long as boxes have same size in training as in prediction
        index_id = info["Inputs"]["Material ID"]["index"]
        index_sdf = info["Inputs"]["SDF"]["index"]
        loc_hp = self.dist_corner_hp
        assert self.inputs[index_id, loc_hp[0], loc_hp[1]] == 1, f"No HP at {self.pos}"
        self.inputs[index_sdf] = SignedDistanceTransform().sdf(self.inputs[index_id].copy(), Tensor(loc_hp))
        assert self.inputs[index_sdf].max() == 1 and self.inputs[index_sdf].min() == 0, "SDF not in [0,1]"
        
    def apply_nn(self, model):
        input = unsqueeze(Tensor(self.inputs), 0)
        model.eval()
        output = model(input)
        output = output.squeeze().detach().numpy()
        return output

    def save(self, run_id:str="", dir:str="HP-Boxes"):
        if not os.path.exists(dir):    
            os.makedirs(f"{dir}/Inputs")
            os.makedirs(f"{dir}/Labels")
        save(self.inputs, f"{dir}/Inputs/{run_id}HP_{self.id}.pt")
        save(self.label, f"{dir}/Labels/{run_id}HP_{self.id}.pt")

    def plot_fields(self, n_subplots:int, domain:Domain):
        plt.subplots(n_subplots, 1, sharex=True,figsize=(20, 3*(n_subplots)))
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

    def plot_1HP(self, domain:Domain, dir:str="HP-Boxes"):
        n_subplots = len(self.inputs) + 1
        self.plot_fields(n_subplots, domain)
        self.plot_prediction_1HP(n_subplots, idx=n_subplots)
        plt.savefig(f"{dir}/hp_{self.id}.png")

    def plot_2HP(self, domain:Domain, dir:str="HP-Boxes_2HP"):
        n_subplots = len(self.inputs) + 2
        self.plot_fields(n_subplots, domain)
        self.plot_prediction_1HP(n_subplots, idx=n_subplots-1)
        self.plot_prediction_2HP(n_subplots, idx=n_subplots)
        plt.savefig(f"{dir}/hp_{self.id}.png")

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

def pipeline(case:str, dataset_large_name:str, model_name_1HP:str, dataset_trained_model_name:str, input_pg:str, model_name_2HP:str=None, device:str="cuda:0"):
    """
    assumptions:
    - 1hp-boxes are generated already
    - network is trained
    - cell sizes of 1hp-boxes and domain are the same
    - boundaries of boxes around at least one hp is within domain
    """
    # prepare large dataset if not done yet
    datasets_raw_domain_dir, datasets_prepared_domain_dir, dataset_domain_path, datasets_model_trained_with_path, model_1hp_path, model_2hp_path, name_extension, datasets_prepared_2hp_dir = set_paths(dataset_large_name, model_name_1HP, dataset_trained_model_name, input_pg, model_name_2hp=model_name_2HP)
    destination_2hp_prep = pathlib.Path(datasets_prepared_2hp_dir,f"{dataset_large_name}_2hp")
    
    # load model from 1hp-NN
    model_1HP = load_model({"model_choice": "unet", "in_channels": 5}, model_1hp_path, "model", device)

    # prepare domain dataset if not yet happened
    if not os.path.exists(dataset_domain_path):
        prepare_dataset(raw_data_directory = datasets_raw_domain_dir,
                        datasets_path = datasets_prepared_domain_dir,
                        dataset_name = dataset_large_name,
                        input_variables = input_pg + "ksio",
                        power2trafo = False,
                        info = load_yaml(datasets_model_trained_with_path, "info"),
                        name_extension = name_extension) # norm with data from dataset that NN was trained with!
    else:
        print(f"Domain {dataset_domain_path} already prepared")

    if case=="2HP apply":
        print(f"2HP-NN already prepared")
        model_2HP = load_model({"model_choice": "unet", "in_channels": 5}, model_2hp_path, "model", device)
        
    # prepare 2HP dataset
    list_runs = os.listdir(os.path.join(dataset_domain_path, "Inputs"))
    for run_file in tqdm(list_runs, desc=case, total=len(list_runs)):
        run_id = f'{run_file.split(".")[0]}_'
        domain, single_hps = box_generation_2HP(run_file, run_id, dataset_domain_path, model_1HP, destination_2hp_prep, case, input_pg)
        if case=="2HP apply":
            # model interaction of 2 heat plumes : apply 2HP-NN to 2HP dataset
            apply_2HP_NN(domain, single_hps, model_2HP, input_pg, plot=True)

def box_generation_2HP(run_file:str, run_id:int, dataset_domain_path:str, model_1HP, destination_2hp_prep:str, case:str, input_pg:str="p", plot:bool=False):
    domain = Domain(dataset_domain_path, stitching_method="max", file_name=run_file)
    # generate 1hp-boxes and extract information like perm and ids etc.
    single_hps = domain.extract_hp_boxes()

    # apply learned NN to predict the heat plumes
    hp : HeatPump
    for hp in single_hps:
        hp.prediction_1HP = hp.apply_nn(model_1HP)
        if case=="1HP": # orig. pipeline
            hp.save(dir="HP-Boxes_1HP")
        # save predicted Temp field as input for training as well
        hp.prediction_1HP = domain.reverse_norm(hp.prediction_1HP, property="Temperature [C]")
        if case=="1HP": # orig. pipeline
            hp.plot_1HP(domain, dir="HP-Boxes_1HP")
        domain.add_hp(hp, hp.prediction_1HP)

    if plot or case in ["1HP", "2HP prepare"]:
        domain.plot("tkio"+input_pg)
    
    if case in ["2HP prepare", "2HP apply"]:
        for hp in single_hps:
            domain.overwrite_boxes_prediction_1HP(hp)
            hp.prediction_1HP = domain.norm(hp.prediction_1HP, property="Temperature [C]")
            hp.inputs[domain.get_index_from_name("Original Temperature [C]")] = hp.prediction_1HP.copy()
            if case=="2HP prepare":
                hp.save(run_id=run_id, dir=destination_2hp_prep)
                logging.info(f"Saved {hp.id} for run {run_id}")
            
        # copy info file
        if case=="2HP prepare":
            save_yaml(domain.info, path=destination_2hp_prep, name_file="info")

    return domain, single_hps

def apply_2HP_NN(domain:Domain, single_hps:list[HeatPump], model_2HP, input_pg:str="p", plot:bool=False):
    # apply learned NN to predict the heat plumes
    domain.prediction_2HP = domain.prediction.copy()
    for hp in single_hps:
        hp.prediction_2HP = hp.apply_nn(model_2HP)
        # save predicted Temp field
        hp.prediction_2HP = domain.reverse_norm(hp.prediction_2HP, property="Temperature [C]")
        domain.add_hp(hp, hp.prediction_2HP)
    if plot:
        domain.plot("tkio"+input_pg)
    
def set_paths(dataset_large_name:str, model_name_1hp:str, dataset_trained_model_name:str, input_pg:str, model_name_2hp:str=None):
    if input_pg == "g":
        name_extension = "_grad_p"
    else:
        name_extension = ""
    if os.path.exists("/scratch/sgs/pelzerja/"):
        # on remote computer: ipvsgpu1
        datasets_raw_domain_dir = "/scratch/sgs/pelzerja/datasets/2hps_demonstrator"
        datasets_prepared_domain_dir = "/home/pelzerja/pelzerja/test_nn/datasets_prepared/2hps_demonstrator"
        models_1hp_dir = "/home/pelzerja/pelzerja/test_nn/1HP_NN/runs"
        datasets_prepared_1hp_dir = "/home/pelzerja/pelzerja/test_nn/datasets_prepared"
    else:
        # on another computer, hopefully on lapsgs29
        datasets_raw_domain_dir = "/home/pelzerja/Development/datasets/2hps_demonstrator"
        datasets_prepared_domain_dir = "/home/pelzerja/Development/datasets_prepared/2hps_demonstrator"
        models_1hp_dir = "/home/pelzerja/Development/models/1HP_NN"
        models_2hp_dir = "/home/pelzerja/Development/models/2HP_NN"
        datasets_prepared_1hp_dir = "/home/pelzerja/Development/datasets_prepared/1HP_NN"
        datasets_prepared_2hp_dir = "/home/pelzerja/Development/datasets_prepared/2HP_NN"

    dataset_domain_path = os.path.join(datasets_prepared_domain_dir, dataset_large_name+name_extension)
    datasets_model_trained_with_path = os.path.join(datasets_prepared_1hp_dir, dataset_trained_model_name)
    model_1hp_path = os.path.join(models_1hp_dir, model_name_1hp)
    model_2hp_path = None
    if model_name_2hp is not None:
        model_2hp_path = os.path.join(models_2hp_dir, model_name_2hp)

    return datasets_raw_domain_dir, datasets_prepared_domain_dir, dataset_domain_path, datasets_model_trained_with_path, model_1hp_path, model_2hp_path, name_extension, datasets_prepared_2hp_dir

def get_box_corners(pos_hp, size_hp_box, distance_hp_corner, domain_shape):
    corner_ll = pos_hp - np.array(distance_hp_corner)                           # corner lower left
    corner_ur = pos_hp + np.array(size_hp_box) - np.array(distance_hp_corner)   # corner upper right

    assert corner_ll[0] >= 0 and corner_ur[0] < domain_shape[0], f"HP BOX at {pos_hp} is with x=({corner_ll[0]}, {corner_ur[0]}) in x-direction (0, {domain_shape[0]}) not in domain"
    assert corner_ll[1] >= 0 and corner_ur[1] < domain_shape[1], f"HP BOX at {pos_hp} is with y=({corner_ll[1]}, {corner_ur[1]}) in y-direction (0, {domain_shape[1]}) not in domain"
    
    return corner_ll, corner_ur


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_large", type=str, default="benchmark_dataset_2d_2hps_iso_perm")
    parser.add_argument("--model", type=str, default="current_unet_benchmark_dataset_2d_100datapoints")
    parser.add_argument("--dataset_boxes", type=str, default="benchmark_dataset_2d_100datapoints")
    parser.add_argument("--input_pg", type=str, default="g")
    parser.add_argument("--model_2hp", type=str, default=None)
    parser.add_argument("--case", type=str)

    args = parser.parse_args()
    args.device = "cpu"

    pipeline(dataset_large_name=args.dataset_large, model_name_1HP=args.model, dataset_trained_model_name=args.dataset_boxes, case=args.case, input_pg=args.input_pg, model_name_2HP=args.model_2hp, device=args.device)
    