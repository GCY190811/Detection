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
        if activation == leaky: activation
    5. maxpool:
        kernel_size
        stride
        if kernel_size == 2 and stride == 1:
            nn.ZerosPad2d()
        nn.maxpool2d()  ???
    6. upsample:
        scale_factor: stride, mode:""

        自己写了一个Upsample的类，只需__init__函数和forward函数。或许pytorch支持，只需一个forward方法，backpropogation部分pytorch内部实现。
    6. route:
        叠加filter_list中的指定filter，route指定的filter???
        模块这里采用了EmptyLayer()
    7. shortcut
        filter同样采用已存在的filer_list中的指定filter.
        模块这里依旧添加层。
    8. yolo部分
        读取anchor, num_classes, img_size: height(只因为是方形？？？)
        添加一个YOLOLayer类。

    nn.ModuleList()
        模型list,只要是nn.Module的子类，都可以添加。子类要求有__init__函数和forward函数
    nn.Sequential()
        一个有序容器，可以放模块，也可以放词典。
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            # ?
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    output_filters=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    # ?
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module(
                    f"batch_norm_{module_i}",
                    nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            # ?
            if kernel_size == 2 and stride == 1:
                modules.add_module(f"_debug_padding_{module_i}",
                                   nn.ZerosPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size,
                                   stride=stride,
                                   padding=int((kernel_size - 1) // 2))
            modules.add_module(f"maxpool_{module_i}", maxpool)

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]),
                                mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            # ?
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]

            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1])
                       for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])
            img_size = hyperparams["height"]

            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module(f"yolo_{module_i}", yolo_layer)

        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """
    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""
    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """Detection layer"""
    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0

    def forward(self, x, targets=None, img_dim=None):

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (x.view(num_samples, self.num_anchors,
                             self.num_classes + 5, grid_size,
                             grid_size).permute(0, 1, 3, 4, 2).contiguous())

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        pred_conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])

        # if grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            # ???
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_bboxes = FloatTensor(prediction[..., :4].shape)
        pred_bboxes[..., 0] = x.data + self.grid_x
        pred_bboxes[..., 1] = y.data + self.grid_y
        pred_bboxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_bboxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat((
            pred_bboxes.view(num_samples, -1, 4) * self.stride,
            pred_conf.view(num_samples, -1, 1),
            pred_cls.view(num_samples, -1, self.num_classes),
        ), -1)

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_bboxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

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
