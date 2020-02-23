from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from utils.parse_config import *

import matplotlib.pyplot as pyplot
import matplotlib.patches as patches


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    1. 单独处理网络的input, 这一部分不算网络的层数中。
    2. 记录通道数，第一个就是input的filter size
       使用nn.ModuleList()
    3. 循环遍历每个模块
       使用nn.Sequential()
    4. convolutional:
        filter
        size
        stride
        pad
        if bn: bn
        else: activation
    """


class Darknet(nn.Module):
    """YOLOV3 object detection model"""
    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.model_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = [
            layer[0] for layer in self.module_list
            if hassttr(layer[0], "metrics")
        ]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)
