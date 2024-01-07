import torch
from models.model_entry import select_model
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

# 加载 .npy 文件为 NumPy 数组
def load_npy_as_ndarray(file_path):
    return np.load(file_path)

def load_match_dict(model, model_path):
    # model: single gpu model, please load dict before warp with nn.DataParallel
    pretrain_dict = torch.load(model_path)
    model_dict = model.state_dict()
    # the pretrain dict may be multi gpus, cleaning
    pretrain_dict = {k.replace('.module', ''): v for k, v in pretrain_dict.items()}
    # 1. filter out unnecessary keys
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if
                       k in model_dict and v.shape == model_dict[k].shape}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrain_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

def set_rand_seed(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def prepare_model(args, is_train: bool = True):
    model = select_model(args)

    set_rand_seed(args)

    if is_train:
        if args.load_model_path != '':
            print("=> using pre-trained weights for DPSNet")
            if args.load_not_strict:
                load_match_dict(model, args.load_model_path)
            else:
                model.load_state_dict(torch.load(args.load_model_path).state_dict())

        model = torch.nn.DataParallel(model)
    else:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(args.load_model_path))
        model.eval()

    return model


def multi_processes_execute(task, data, workers=None, use_tqdm=True):
    with Pool(processes=workers) as pool:
        data_ret = pool.imap(task, data)
        if use_tqdm:
            data_ret = tqdm(data_ret, total=len(data))
        return list(data_ret)

