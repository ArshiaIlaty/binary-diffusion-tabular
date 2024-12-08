from typing import Literal, Union, List, Dict
from pathlib import Path
import yaml

import pandas as pd

import torch


__all__ = [
    "TASK",
    "exists",
    "default",
    "PathOrStr",
    "cycle",
    "zero_out_randomly",
    "get_base_model",
    "get_config",
    "save_config",
]


TASK = Literal["classification", "regression"]

PathOrStr = Union[str, Path]


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def zero_out_randomly(
    tensor: torch.Tensor, probability: float, dim: int = 0
) -> torch.Tensor:
    """Zero out randomly selected elements of a tensor with a given probability at a given dimension

    Args:
        tensor: tensor to zero out
        probability: probability of zeroing out an element
        dim: dimension along which to zero out elements

    Returns:
        torch.Tensor: tensor with randomly zeroed out elements
    """

    mask = torch.rand(tensor.shape[dim]) < probability
    tensor[mask] = 0
    return tensor


def get_base_model(model):
    if hasattr(model, "module"):
        return model.module
    return model


def get_config(config):
    with open(config, "r") as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def save_config(config: Dict, yaml_file_path: PathOrStr) -> None:
    """save config to yaml file

    Args:
        config: config to save
        yaml_file_path: path to yaml file
    """

    try:
        with open(yaml_file_path, "w") as file:
            yaml.dump(config, file, sort_keys=False, default_flow_style=False)
    except Exception as e:
        print(f"Error saving dictionary to YAML file: {e}")
