from models.vgg_model import VGG
from models.cnn_model import CNN
from models.exquisitenetv2 import ExquisiteNetV2
from models.classifier.mlp import MLP
import torch.nn as nn


def select_model(args):
    type2model = {
        'vgg': VGG(args),
        'cnn': nn.Sequential(
             CNN(include_fc=True, hidden_dim=300),
             MLP(300, 300, 10, 0.3, depth=1)
        ),
        'ExquisiteNetV2': ExquisiteNetV2(10, 10),
    }
    
    model = type2model[args.model_type]
    return model


def equip_multi_gpu(model, args):
    model = nn.DataParallel(model, device_ids=args.gpus)
    return model