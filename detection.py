from pathlib import Path

import torch

from models.yolo import Model
from utils.general import set_logging

dependencies = ['torch', 'yaml']
set_logging()


def custom(path_or_model='path/to/model.pt', autoshape=True, device='cpu'):
    """YOLOv5-custom model from https://github.com/ultralytics/yolov5

    Arguments (3 options):
        path_or_model (str): 'path/to/model.pt'
        path_or_model (dict): torch.load('path/to/model.pt')
        path_or_model (nn.Module): torch.load('path/to/model.pt')['model']

    Returns:
        pytorch model
    """
    if device == 'cpu':
        model = torch.load(path_or_model, map_location=torch.device('cpu')) if isinstance(path_or_model, str) else path_or_model  # load checkpoint
    else:
        model = torch.load(path_or_model) if isinstance(path_or_model, str) else path_or_model  # load checkpoint

    if isinstance(model, dict):
        model = model['model']  # load model

    hub_model = Model(model.yaml).to(next(model.parameters()).device)  # create
    hub_model.load_state_dict(model.float().state_dict())  # load state_dict
    hub_model.names = model.names  # class names
    return hub_model.autoshape() if autoshape else hub_model