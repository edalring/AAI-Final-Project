import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from models.vgg_model import VGG
from dataloader import MNISTDataset 
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable

from utils import logger
from utils import torch_utils
import options





class Trainer:
    def __init__(self, model, criterion, train_loader, val_loader, args, optimizer=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = args.epochs
        self.args = args

        self.logger = logger.Logger(args)

        # Adam optimizer is the default choice, which is the best choice in most cases
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr,
                                          betas=(self.args.momentum, self.args.beta),
                                          weight_decay=self.args.weight_decay) \
                        if self.optimizer is None else self.optimizer

    def train(self):
        for epoch in range(self.args.epochs):
            # train for one epoch
            self.train_per_epoch(epoch)
            self.val_per_epoch(epoch)
            self.logger.save_curves(epoch)
            self.logger.save_check_point(self.model, epoch)
            self.print_per_epoch(epoch)

    def train_per_epoch(self, epoch):
        # switch to train mode
        self.model.train()

        for i, data in enumerate(self.train_loader):
            img, pred, label = self.step(data)

            # compute loss
            metrics = self.compute_metrics(pred, label, is_train=True)

            # get the item for backward
            loss = metrics['train/loss']

            # compute gradient and do Adam step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # logger record
            for key, val in metrics.items():
                self.logger.record_scalar(key, val)

            # # only save img at first step
            # if i == len(self.train_loader) - 1:
            #     self.logger.save_imgs(self.gen_imgs_to_write(img, pred, label, True), epoch)

            # # monitor training progress
            if i % self.args.print_freq == 0:
                print('Train: Epoch {}/{} batch {} Loss {}'.format(epoch, self.epochs, i, loss))



    def val_per_epoch(self, epoch):
        self.model.eval()
        for i, data in enumerate(self.val_loader):
            img, pred, label = self.step(data)
            metrics = self.compute_metrics(pred, label, is_train=False)

            for key in metrics.keys():
                self.logger.record_scalar(key, metrics[key])

            # if i == len(self.val_loader) - 1:
            #     self.logger.save_imgs(self.gen_imgs_to_write(img, pred, label, False), epoch)


    def print_per_epoch(self, epoch):
        avg_metric = self.logger.get_average_metric()


        average_train_loss = avg_metric['train/loss']
        train_accuracy     = avg_metric['train/acc']
        average_val_loss   = avg_metric['val/loss']
        val_accuracy     = avg_metric['val/acc']

        width = int((np.log10(self.epochs) + 1))
        print(f'Epoch {epoch+1:{width}}/{self.epochs} '
            f'- Train Loss: {average_train_loss:10.8f}\tTrain Acc: {train_accuracy:.2%}'
            f'\t- Test Loss: {average_val_loss:10.8f}\tTest Acc: {val_accuracy:.2%}')


    def step(self, data):
        img, label = data
        # warp input
        img = img.to(self.device)
        label = label.to(self.device)

        # compute output
        pred = self.model(img).to(self.device)
        return img, pred, label

    def compute_metrics(self, pred, label, is_train):
        # you can call functions in metrics.py
        # l1 = (pred - gt).abs().mean()
        prefix = 'train/' if is_train else 'val/'
        
        loss    = self.criterion(pred, label)
        cnt_hit = torch.sum(torch.argmax(pred, dim=1) == label).item()
        acc     = cnt_hit / len(label)

        metrics = {
            prefix + 'loss': loss,
            prefix + 'acc': acc,
        }

        return metrics

    def gen_imgs_to_write(self, img, pred, label, is_train):
        # override this method according to your visualization
        prefix = 'train/' if is_train else 'val/'
        return {
            prefix + 'img': img[0][0].resize(28, 28, 1),
            prefix + 'pred': pred[0],
            prefix + 'label': label[0]
        }

    def compute_loss(self, pred, label):
        if self.args.loss == 'l1':
            loss = (pred - label).abs().mean()
        elif self.args.loss == 'ce':
            loss = torch.nn.functional.cross_entropy(pred, label)
        else:
            loss = torch.nn.functional.mse_loss(pred, label)
        return loss


def main():
    args = options.prepare_train_args()
    model = torch_utils.prepare_model(args)

    critirion = nn.CrossEntropyLoss()
    
    data_path ='processed_data'
    train_dataset = MNISTDataset(data_path=data_path, mode='train')
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    val_dataset = MNISTDataset(data_path=data_path, mode='val')
    validloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)


    trainer = Trainer(model=model, criterion=critirion, train_loader=trainloader, val_loader=validloader, args=args)
    trainer.train()


if __name__ == '__main__':
    main()