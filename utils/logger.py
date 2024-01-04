from torch.utils.tensorboard import SummaryWriter
import os
import torch


class Recoder:
    def __init__(self):
        self.metrics = {}
        self.avg_metrics = {}

    def record(self, name, value):
        if name in self.metrics.keys():
            self.metrics[name].append(value)
        else:
            self.metrics[name] = [value]

    def summary(self):
        
        for key, values in self.metrics.items():
            self.avg_metrics[key] = sum(values) / len(values)
            del self.metrics[key][:]
            self.metrics[key] = []
        return self.avg_metrics 


class Logger:
    def __init__(self, args):
        self.writer = SummaryWriter(log_dir=args.model_dir)
        self.recoder = Recoder()
        self.model_dir = args.model_dir

    def tensor2img(self, tensor):
        # implement according to your data, for example call viz.py
        return tensor.cpu().numpy()

    def record_scalar(self, name, value):
        self.recoder.record(name, value)

    def save_curves(self, epoch):
        kvs = self.recoder.summary()
        for key, val in kvs.items():
            self.writer.add_scalar(key, val, epoch)

    def get_average_metric(self):
        return self.recoder.avg_metrics

    def save_imgs(self, names2imgs: dict, epoch: int):
        for name, img_data in names2imgs.items():
            self.writer.add_image(name, self.tensor2img(img_data), epoch)

    def save_check_point(self, model, epoch, step=0):
        model_name = '{epoch:02d}_{step:06d}.pth'.format(epoch=epoch, step=step)
        path = os.path.join(self.model_dir, model_name)
        # don't save model, which depends on python path
        # save model state dict
        torch.save(model.state_dict(), path)