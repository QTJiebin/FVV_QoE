# function: train the QA model for FVVs
# usage: pls refer to "__main__" below
# author: Jiebin Yan
# email: jiebinyan@foxmail.com
# v1.0.0

import torch
import argparse
from tqdm import tqdm
from VSFA import VSFA
import torch.optim as optim
from torch.autograd import Variable
from scipy import stats
import os
import scipy.io
import random
import pandas as pd
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

from folders import BTFVVFolder, BTFVVFolderVSFA
from tools import mkDir
from tools import str2bool


class TrainManager(object):
    """A LSTM QA model for FVVs"""
    def __init__(self, data, options):
        self._data = data
        self._options = options

        self._para_saving_path = self._data['para_saving_path']
        mkDir(self._para_saving_path)

        if torch.cuda.is_available():
            self._device = torch.device("cuda:0")
        else:
            self._device = torch.device('cpu')

        self._net = VSFA()
        self._net.to(self._device)
        self._loss = nn.L1Loss()
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self._options['base_lr'],
                                           weight_decay=self._options['weight_decay'])
        self._lr_scheduler = optim.lr_scheduler.StepLR(self._optimizer, self._options['step_size'], self._options['gamma'])

        # # vsfa
        if self._options['vsfa']:
            print("running by vsfa")
            self._train_data = BTFVVFolderVSFA(self._data['data_path'], self._data['vsfa_feature_path'],
                                               self._options['iter_number'], self._options['video_max_number'], True)
            self._train_data_loader = torch.utils.data.DataLoader(self._train_data, batch_size=self._options["batch_size"],
                                                                  shuffle=True)
            self._test_data = BTFVVFolderVSFA(self._data['data_path'], self._data['vsfa_feature_path'],
                                          self._options['iter_number'], self._options['video_max_number'], False)
            self._test_data_loader = torch.utils.data.DataLoader(self._test_data, 1, shuffle=False)

        else:  # # normal sampling strategies
            print("running by others")
            self._train_data = BTFVVFolder(self._data['data_path'], self._data['vsfa_feature_path'],
                                           self._options['iter_number'], self._options['frame_number'], True)
            self._train_data_loader = torch.utils.data.DataLoader(self._train_data, batch_size=self._options["batch_size"],
                                                                  shuffle=True)
            self._test_data = BTFVVFolder(self._data['data_path'], self._data['vsfa_feature_path'],
                                          self._options['iter_number'], self._options['frame_number'], False)
            self._test_data_loader = torch.utils.data.DataLoader(self._test_data, 1, shuffle=False)
        self._performance_file_name = self._data['per_save_file']

    def train(self):

        performance_saving = open(self._performance_file_name, 'a')
        total_epoches = self._options['epochs']
        best_srcc = 0
        model_path = os.path.join(self._data['para_saving_path'], (str(self._options['iter_number']) + '.pkl'))
        
        if not self._options['vsfa']:
            performance_saving.write("frame number:\t")
            performance_saving.write(str(self._options['frame_number']) + ",\t")
            performance_saving.write('\n')        

        for epoch in tqdm(range(total_epoches)):

            self._net.train()
            self._lr_scheduler.step()

            for step, (features, mos, frames) in enumerate(self._train_data_loader):

                features = Variable(features)
                features = features.to(self._device)
                mos = Variable(mos)
                mos = mos.to(self._device)

                predictions = self._net(features.float(), frames.float())
                loss = self._loss(mos, predictions)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                if step % 10 == 0:
                    print("epoch: {}, step: {}, loss: {}".format(epoch, step, loss))

            # # test per epoch
            plcc, srcc, rmse = self.evaluate()

            if abs(srcc) > best_srcc:
                best_srcc = srcc
                torch.save(self._net.state_dict(), model_path)

            print("plcc: {}, scrr: {}, rmse: {}".format(plcc, srcc, rmse))
            performance_saving.write("epoch:\t")
            performance_saving.write(str(epoch) + ",\t")
            performance_saving.write("plcc:\t")
            performance_saving.write(str(plcc) + ",\t")
            performance_saving.write("srcc:\t")
            performance_saving.write(str(srcc) + ",\t")
            performance_saving.write("rmse:\t")
            performance_saving.write(str(rmse) + ",\t")
            performance_saving.write('\n')

        performance_saving.close()
        print("well done...")

    def evaluate(self):

        self._net.eval()

        mos_matrix = []
        pre_matrix = []
        for step, (features, mos, frames) in enumerate(self._test_data_loader):
            mos_matrix.append(mos.cpu().tolist()[0])

            features = Variable(features)
            features = features.to(self._device)

            if self._options['vsfa']:
                predictions = self._net(features.float(), frames.float())
                predictions = predictions.cpu().data.numpy()
            else:
                features = torch.squeeze(features, dim=0)
                frames = torch.squeeze(frames, dim=0)
                predictions = self._net(features.float(), frames.float())
                predictions = predictions.cpu().data.numpy()

            mean_predictions = np.mean(predictions)
            pre_matrix.append(mean_predictions)

        mos_matrix = np.array(mos_matrix)
        pre_matrix = np.array(pre_matrix)

        plcc, _ = stats.pearsonr(pre_matrix, mos_matrix)
        srcc, _ = stats.spearmanr(pre_matrix, mos_matrix)
        rmse = np.sqrt(((pre_matrix - mos_matrix) ** 2).mean())

        return round(plcc, 4), round(srcc, 4), round(rmse, 4)


def demo(model, iterm):
    """predict a QoE score given a FVV"""
    pass
    """waiting for a moment"""


def main(data, options):
    """start training"""
    trainer = TrainManager(data, options)
    trainer.train()


if __name__ == '__main__':
    """The main function."""

    parser = argparse.ArgumentParser(
        description='Train CNN for DIBR Images.')

    parser.add_argument('--data_path', dest='data_path', type=str,
                        default="./Overall_Mos", help='The Path for Saving Training and Testing Data (Video Name and Mos).')
    parser.add_argument('--vsfa_feature_path', dest='vsfa_feature_path', type=str,
                        default="./VSFA_Features", help='The Path of VSFA Features.')
    parser.add_argument('--vsfa', dest='vsfa', type=str2bool,
                        default=True, help='Truth represents train VSFA, while others denote other options.')
    parser.add_argument('--para_saving_path', dest='para_saving_path', type=str,
                        default="./Model_Savings", help='Network Parameter Saving Path.')
    # parser.add_argument('--per_save_file', dest='per_save_file', type=str,
    #                     default="performance_4.txt", help='Performance Saving File.')
    parser.add_argument('--base_lr', dest='base_lr', type=float, default=1e-4,
                        help='Base learning rate for training')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=30, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int,
                        default=100, help='Epochs for training')
    parser.add_argument('--iter_number', dest='iter_number', type=int,
                        default=6, help='Inters for training')
    parser.add_argument('--video_max_number', dest='video_max_number', type=int,
                        default=300, help='A hyper parameter for training VSFA')
    parser.add_argument('--frame_number', dest='frame_number', type=int,
                        default=16, help='Number of input frames')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        default=5e-4, help='Weight decay')

    parser.add_argument('--step_size', dest='step_size', type=int,
                        default=20, help='Step_size - learning rate')
    parser.add_argument('--gamma', dest='gamma', type=float,
                        default=0.1, help='Gamma - learning rate')

    args = parser.parse_args()
    if args.base_lr <= 0:
        raise AttributeError('--base_lr parameter must >0.')
    if args.batch_size <= 0:
        raise AttributeError('--batch_size parameter must >0.')
    if args.epochs < 0:
        raise AttributeError('--epochs parameter must >=0.')
    if args.weight_decay <= 0:
        raise AttributeError('--weight_decay parameter must >0.')

    data = {
        "data_path": args.data_path,
        "vsfa_feature_path": args.vsfa_feature_path,
        "para_saving_path": args.para_saving_path,
        "per_save_file": None
    }

    if args.vsfa:
        data['para_saving_path'] = args.para_saving_path + "_VSFA"
    else:
        data['para_saving_path'] = args.para_saving_path + "_Others"
    data['per_save_file'] = os.path.join(data['para_saving_path'], 'performance_'+str(args.iter_number)+'.txt')

    options = {
        'video_max_number': args.video_max_number,
        'vsfa': args.vsfa,
        'base_lr': args.base_lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        "iter_number": args.iter_number,
        "frame_number": args.frame_number,
        "step_size": args.step_size,
        "gamma": args.gamma
    }

    main(data, options)

    print("good luck ...")























