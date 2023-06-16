import os
import sys
import argparse
import logging
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
        self.prediction_1HP: np.ndarray            = np.ones(self.size) * self.background_temperature
        self.prediction_2HP: np.ndarray            = np.ones(self.size) * self.background_temperature
        self.stitching:Stitching            = Stitching(stitching_method, self.background_temperature)
        # for var in ["SDF", "Material ID [-]", "Original Temperature [C]", "Liquid Pressure [Pa]", "Pressure Gradient [-]", "Permeability X [m^2]"]:
        #     try:
        #         print(f"{var} in {self.get_input_field_from_name(var).max(), self.get_input_field_from_name(var).min()}")
        #     except:
        #         print(f"{var} not in inputs")
        assert self.get_input_field_from_name('Permeability X [m^2]').max() <= 1 and self.get_input_field_from_name('Permeability X [m^2]').min() >= 0, "Permeability X [m^2] not in range (0,1)"
        try:
            assert self.get_input_field_from_name('Pressure Gradient [-]').max() <= 1 and self.get_input_field_from_name('Pressure Gradient [-]').min() >= 0, "Pressure Gradient [-] not in range (0,1)"
        except:
            print(f"Pressure: {self.get_input_field_from_name('Liquid Pressure [Pa]').max(), self.get_input_field_from_name('Liquid Pressure [Pa]').min()}")
            # assert self.get_input_field_from_name('Liquid Pressure [Pa]').max() <= 1 and self.get_input_field_from_name('Liquid Pressure [Pa]').min() >= 0, "Liquid Pressure [Pa] not in range (0,1)"
        
    def load_datapoint(self, dataset_domain_path:str, case:str = "Inputs", file_name = "RUN_0.pt"):
        # load dataset of large domain
        file_path = os.path.join(dataset_domain_path, case, file_name)
        data = load(file_path).detach().numpy()
        return data
    
    def get_index_from_name(self, name:str):
        return self.info["Inputs"][name]["index"]
    
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
            # if corner_ll[0] < 0 or corner_ur[0] >= self.inputs.shape[1]:
            #     print(f"HP BOX at {pos_hp} is with ({corner_ll[0]}, {corner_ur[0]}) in x-direction (0, {self.inputs.shape[1]}) not in domain")
            #     continue
            # if corner_ll[1] < 0 or corner_ur[1] >= self.inputs.shape[2]:
            #     print(f"HP BOX at {pos_hp} is with ({corner_ll[1]}, {corner_ur[1]}) in y-direction (0, {self.inputs.shape[2]}) not in domain")
            #     continue
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
                
    def add_hp(self, hp: "HeatPump"):
        # compose learned fields into large domain with list of ids, pos, orientations
        for i in range(hp.prediction_1HP.shape[0]):
            for j in range(hp.prediction_1HP.shape[1]):
                x,y = self.coord_trafo(hp.pos, (i-hp.dist_corner_hp[0],j-hp.dist_corner_hp[1]), hp.orientation)
                if 0 <= x < self.prediction_1HP.shape[0] and 0 <= y < self.prediction_1HP.shape[1]:
                    self.prediction_1HP[x,y] = self.stitching(self.prediction_1HP[x, y], hp.prediction_1HP[i, j])

    def overwrite_boxes_prediction_1HP(self, hp: "HeatPump"):
        # overwrite hp.prediction_1HP (originally only information from 1HP) with overlapping predicted temperature fields from domain.prediction_1HP
        # therefor extract self.predictions_1HP in area of hp
        corner_ll, corner_ur = get_box_corners(hp.pos, hp.prediction_1HP.shape, hp.dist_corner_hp, self.prediction_1HP.shape)
        field = self.prediction_1HP[corner_ll[0]:corner_ur[0], corner_ll[1]:corner_ur[1]].copy()
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
                plt.imshow(self.prediction_1HP.T)
                plt.gca().invert_yaxis()
                plt.xlabel("y [cells]")
                plt.ylabel("x [cells]")
                _aligned_colorbar(label=f"Predicted {property}")
                idx+=1
                plt.subplot(n_subplots, 1, idx)
                self.label = self.reverse_norm(self.label, property)
                plt.imshow(abs(self.prediction_1HP.T-np.squeeze(self.label.T)))
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
        self.prediction_1HP = output

    def save(self, run_id = "", dir:str = "HP-Boxes"):
        if not os.path.exists(dir):    
            os.makedirs(f"{dir}/Inputs")
            os.makedirs(f"{dir}/Labels")
        save(self.inputs, f"{dir}/Inputs/{run_id}HP_{self.id}.pt")
        save(self.label, f"{dir}/Labels/{run_id}HP_{self.id}.pt")

    def plot(self):
        n_subplots = len(self.inputs) + 1
        plt.subplots(n_subplots, 1, sharex=True,figsize=(20, 3*(n_subplots)))
        idx = 1
        for input in self.inputs:
            plt.subplot(n_subplots, 1, idx)
            plt.imshow(input.T)
            plt.gca().invert_yaxis()
            plt.xlabel("y [cells]")
            plt.ylabel("x [cells]")
            _aligned_colorbar()
            idx += 1
        plt.subplot(n_subplots, 1, idx)
        plt.imshow(self.prediction_1HP.T)
        plt.gca().invert_yaxis()
        plt.xlabel("y [cells]")
        plt.ylabel("x [cells]")
        _aligned_colorbar(label="Temperature [C]")
        dir = "HP-Boxes"
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

def pipeline(dataset_large_name:str, model_name:str, dataset_trained_model_name:str, input_pg:str, device:str="cuda:0"):
    """
    assumptions:
    - 1hp-boxes are generated already
    - network is trained
    - cell sizes of 1hp-boxes and domain are the same
    - boundaries of boxes around hps are within domain
    """
    # prepare large dataset if not done yet
    datasets_raw_domain_dir, datasets_prepared_domain_dir, dataset_domain_path, datasets_model_trained_with_path, model_path, name_extension, _ = set_paths(dataset_large_name, model_name, dataset_trained_model_name, input_pg)
    
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
        
    domain = Domain(dataset_domain_path, stitching_method="max")
    # generate 1hp-boxes and extract information like perm and ids+pos+orientations-list, BEFORE: choose 1hp-boxes
    # TODO v-orientation currently not possible -> p-orientation for now? or kick out orientation and work with square boxes?
    single_hps = domain.extract_hp_boxes()

    # apply learned NN to predict the heat plumes
    model = load_model({"model_choice": "unet", "in_channels": 5}, model_path, "model", device)
    hp : HeatPump
    for hp in single_hps:
        hp.apply_nn(model)
        hp.save()
        hp.prediction_1HP = domain.reverse_norm(hp.prediction_1HP, property="Temperature [C]")
        hp.plot()
        domain.add_hp(hp)
    domain.plot("tkio"+input_pg)
    
    #TODO LATER: smooth large domain and extend heat plumes

def prep_data_2hp_NN(dataset_large_name:str, model_name_1HP:str, dataset_trained_model_name:str, input_pg:str, device:str="cuda:0"):
    """
    assumptions:
    - 1hp-boxes are generated already
    - network is trained
    - cell sizes of 1hp-boxes and domain are the same
    - boundaries of boxes around at least one hp is within domain
    """
    # prepare large dataset if not done yet
    datasets_raw_domain_dir, datasets_prepared_domain_dir, dataset_domain_path, datasets_model_trained_with_path, model_path, name_extension, datasets_prepared_2hp_dir = set_paths(dataset_large_name, model_name_1HP, dataset_trained_model_name, input_pg)
    destination_2hp_prep = datasets_prepared_2hp_dir + f"/{dataset_large_name}_2hp"
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

    # load model from 1hp-NN
    model = load_model({"model_choice": "unet", "in_channels": 5}, model_path, "model", device)

    # for file in dataset_domain_path:
    for run_file in os.listdir(os.path.join(dataset_domain_path, "Inputs")):
        run_id = f'{run_file.split(".")[0]}_'
            
        domain = Domain(dataset_domain_path, stitching_method="max", file_name=run_file)
        # generate 1hp-boxes and extract information like perm and ids etc.
        single_hps = domain.extract_hp_boxes()

        # apply learned NN to predict the heat plumes
        hp : HeatPump
        for hp in single_hps:
            hp.apply_nn(model)
            # save predicted Temp field as input for training as well
            hp.prediction_1HP = domain.reverse_norm(hp.prediction_1HP, property="Temperature [C]")
            domain.add_hp(hp)
        domain.plot("tkio"+input_pg)
        
        for hp in single_hps:
            domain.overwrite_boxes_prediction_1HP(hp)
            hp.prediction_1HP = domain.norm(hp.prediction_1HP, property="Temperature [C]")
            hp.inputs[domain.get_index_from_name("Original Temperature [C]")] = hp.prediction_1HP
            hp.save(run_id=run_id, dir=destination_2hp_prep)
            logging.info(f"Saved {hp.id} for run {run_id}")

        # copy info file
        save_yaml(domain.info, path=destination_2hp_prep, name_file="info")

def set_paths(dataset_large_name:str, model_name:str, dataset_trained_model_name:str, input_pg:str):
    if input_pg == "g":
        name_extension = "_grad_p"
    else:
        name_extension = ""
    if os.path.exists("/scratch/sgs/pelzerja/"):
        # on remote computer: ipvsgpu1
        datasets_raw_domain_dir = "/scratch/sgs/pelzerja/datasets/2hps_demonstrator"
        datasets_prepared_domain_dir = "/home/pelzerja/pelzerja/test_nn/datasets_prepared/2hps_demonstrator"
        models_dir = "/home/pelzerja/pelzerja/test_nn/1HP_NN/runs"
        datasets_prepared_1hp_dir = "/home/pelzerja/pelzerja/test_nn/datasets_prepared"
    else:
        # on another computer, hopefully on lapsgs29
        datasets_raw_domain_dir = "/home/pelzerja/Development/datasets/2hps_demonstrator"
        datasets_prepared_domain_dir = "/home/pelzerja/Development/datasets_prepared/2hps_demonstrator"
        models_dir = "/home/pelzerja/Development/1HP_NN/runs"
        datasets_prepared_1hp_dir = "/home/pelzerja/Development/datasets_prepared/1HP_NN"
        datasets_prepared_2hp_dir = "/home/pelzerja/Development/datasets_prepared/2HP_NN"

    dataset_domain_path = os.path.join(datasets_prepared_domain_dir, dataset_large_name+name_extension)
    datasets_model_trained_with_path = os.path.join(datasets_prepared_1hp_dir, dataset_trained_model_name)
    model_path = os.path.join(models_dir, model_name)

    return datasets_raw_domain_dir, datasets_prepared_domain_dir, dataset_domain_path, datasets_model_trained_with_path, model_path, name_extension, datasets_prepared_2hp_dir

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
    args = parser.parse_args()
    # if args.input_pg == "g":
    #     args.model += "_grad_p"
    #     args.dataset_boxes += "_grad_p"
    args.device = "cpu"

    # pipeline(dataset_large_name=args.dataset_large, model_name=args.model, dataset_trained_model_name=args.dataset_boxes, device=args.device, input_pg=args.input_pg)
    prep_data_2hp_NN(dataset_large_name=args.dataset_large, model_name_1HP=args.model, dataset_trained_model_name=args.dataset_boxes, device=args.device, input_pg=args.input_pg)