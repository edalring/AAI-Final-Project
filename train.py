import json
from torch.utils.data import RandomSampler
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
from test import Tester

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
        self.batch_size = args.batch_size
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
            f'\t- Valid Loss: {average_val_loss:10.8f}\tValid Acc: {val_accuracy:.2%}')


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

class DROTrainer(Trainer):
    def train_per_epoch(self, epoch):
        # switch to train mode
        self.model.train()

        for i, batches in enumerate(zip(*self.train_loader)):
            X = torch.cat([batch[0] for batch in batches], dim=0)
            y = torch.cat([batch[1] for batch in batches], dim=0)
            img, pred, label = self.step((X, y))
            metrics = self.compute_metrics(pred, label, is_train=True)

            
            # get the item for backward
            loss = metrics['train/wst_loss']

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
    
    # # TODO: figure out should we really need this override method? We can directly use the provided valid set
    # # FIX: use the val_loader directly, since the valid set is not divide. Only need to calculate the avg_loss
    # #      and avg_acc
    def val_per_epoch(self, epoch):
        self.model.eval()
        for i, data in enumerate(self.val_loader):
            img, pred, label = self.step(data)
            metrics = self.compute_metrics(pred, label, is_train=False)

            
            # get the item for backward
            # loss = metrics['train/wst_loss']

            # logger record
            for key, val in metrics.items():
                self.logger.record_scalar(key, val)

            # # only save img at first step
            # if i == len(self.train_loader) - 1:
            #     self.logger.save_imgs(self.gen_imgs_to_write(img, pred, label, True), epoch)


    def compute_metrics(self, pred, label, is_train):
        # you can call functions in metrics.py
        # l1 = (pred - gt).abs().mean()

        if is_train:
            prefix = 'train/'
            
            loss    = self.criterion(pred, label)
            cnt_hit = torch.sum(torch.argmax(pred, dim=1) == label).item()
            acc     = cnt_hit / len(label)

            pred_dim = pred.shape[1]
            # split by window size self.batch_size
            labels = label.view(-1, self.batch_size)
            preds  = pred.view(-1, self.batch_size, pred_dim)

            group_size = preds.shape[0]

            losses = torch.zeros(group_size)
            cnt_hits = torch.zeros(group_size)
            for i in range(group_size):
                losses[i] = self.criterion(preds[i], labels[i])
                cnt_hits[i] = torch.sum(torch.argmax(preds[i], dim=1) == label[i]).item() / self.batch_size
            
            worst_loss = torch.max(losses)
            worst_acc  = torch.min(cnt_hits)
            


            metrics = {
                prefix + 'avg_loss': loss,
                prefix + 'avg_acc': acc,
                prefix + 'wst_loss': worst_loss,
                prefix + 'wst_acc': worst_acc,
            }

            return metrics
        else:
            prefix = 'val/'
        
            loss    = self.criterion(pred, label)
            cnt_hit = torch.sum(torch.argmax(pred, dim=1) == label).item()
            acc     = cnt_hit / len(label)

            metrics = {
                prefix + 'avg_loss': loss,
                prefix + 'avg_acc': acc,
            }

            return metrics
    

    def print_per_epoch(self, epoch):
        avg_metric = self.logger.get_average_metric()

        print(avg_metric)
        average_train_loss = avg_metric['train/avg_loss']
        train_accuracy     = avg_metric['train/avg_acc']
        average_val_loss   = avg_metric['val/avg_loss']
        val_accuracy       = avg_metric['val/avg_acc']

        # TODO: There is no wst_loss and wst/acc, maybe we don't need it?
        # average_worst_train_loss = avg_metric['train/wst_loss']
        # worst_train_accuracy     = avg_metric['train/wst_acc']
        # average_worst_val_loss   = avg_metric['val/wst_loss']
        # worst_val_accuracy       = avg_metric['val/wst_acc']

        width = int((np.log10(self.epochs) + 1))
        print(f'Epoch {epoch+1:{width}}/{self.epochs} '
            f'-       Train Loss: {average_train_loss:10.8f}\t'
            f'      Train Acc: {train_accuracy:.2%}\t'
            f'-       Valid Loss: {average_val_loss:10.8f}\t'
            f'      Valid Acc: {val_accuracy:.2%}')
        # print(f'{"":<{width*2 + 8}}'
        #     f'- Worst Train Loss: {average_worst_train_loss:10.8f} \t'
        #     f'Worst Train Acc: {worst_train_accuracy:.2%}\t'
        #     f'- Worst Valid Loss: {average_worst_val_loss:10.8f}\t'
        #     f'Worst Valid Acc: {worst_val_accuracy:.2%}')
        
def get_train_loaders(path, batch_size, idx_table):
    train_loaders = []
    for label_idx in range(10):
        for channel_idx in range(10):
            train_loaders.append(create_dataloader_by_idx(path, int(batch_size//100), idx_table[str(label_idx)][str(channel_idx)]))
    return train_loaders


def create_dataloader_by_idx(path, batch_size, idxs):
    dataset = MNISTDataset(data_path=path, mode='train', idxs=idxs)
    random_sampler = RandomSampler(dataset)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True, sampler=random_sampler)

def main():
    args = options.prepare_train_args()
    model = torch_utils.prepare_model(args)

    critirion = nn.CrossEntropyLoss()
   
    data_path = args.data_path

    if args.straight_forward:

        train_dataset = MNISTDataset(data_path=data_path, mode='train')
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

        val_dataset = MNISTDataset(data_path=data_path, mode='val')
        validloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)


        trainer = Trainer(model=model, criterion=critirion, train_loader=trainloader, val_loader=validloader, args=args)
    else:
        # TODO: fix this
        # FIX: train step for diiferent envs:
        #      1. load the idx infomation from json file
        #      2. prepare train loaders: totally 10(class num)*10(env num) = 100 dataloader
        #      3. prepare validloader: since the valid set is small, set batch_size = length of valid set
        with open('utils/data.json', 'r') as json_file:
            idx_table = json.load(json_file)
        trainloader = get_train_loaders(data_path, args.batch_size, idx_table)
        val_dataset = MNISTDataset(data_path=data_path, mode='val', idxs=None)
        validloader = torch.utils.data.DataLoader(val_dataset, len(val_dataset), shuffle=True, pin_memory=True)
        trainer = DROTrainer(model=model, criterion=critirion, train_loader=trainloader, val_loader=validloader, args=args)

        

    trainer.train()

    test_dataset = MNISTDataset(data_path=args.data_path, mode='test')
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=200, shuffle=True, pin_memory=True)
    tester = Tester(args, model, testloader)
    tester.test()
    print('test_pass!')


if __name__ == '__main__':
    main()