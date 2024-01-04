from models.vgg_model import VGG
import torch.nn as nn


def select_model(args):
    type2model = {
        'vgg': VGG(args),
    }
    
    model = type2model[args.model_type]
    return model


def equip_multi_gpu(model, args):
    model = nn.DataParallel(model, device_ids=args.gpus)
    return model