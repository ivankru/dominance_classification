import yaml
from typing import Any

def parse_config(path: str) -> Any:
    '''
    Function to read the configuration file
    path - path to the config file

    return: data from the config file
    '''
    with open(path, "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)

    return data
