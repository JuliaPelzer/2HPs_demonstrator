import yaml

def load_info(path):
    with open(path.joinpath("info.yaml"), "r") as f:
        info = yaml.safe_load(f)
    return info