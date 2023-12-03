"""File containing the definition of some utility functions"""
from torchvision import transforms
import yaml

def get_normalization_transform():
    """Gets a normalization transform"""
    data_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return data_transform

def load_yaml_config(yaml_file_path):
    """Loads a yaml config file and returns its associated dictionary"""
    with open(yaml_file_path, "r",encoding="utf-8") as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    return config_data
