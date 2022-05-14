# function: provide some useful tools
# author: Jiebin Yan
# email: jiebinyan@foxmail.com
# v1.0.0

from PIL import Image
import numpy as np
import cv2 as cv
import os
import os.path
# from ignite.metrics.metric import Metric
from scipy import stats
import torch
import torchvision.models as models
import torch.nn as nn
import random
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentError('Bollean value expected.')


def getFileName(path, suffix):
    """get file name with the suffix name specified in the catalog"""
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename


def pilLoader(path):
    """read image"""
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def mkDir(path):
    """make dir if not exists"""
    if not os.path.exists(path):
        os.makedirs(path)



if __name__ == '__main__':
    """
        test these pre-defined functions
    """

    print("good luck")
