from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import logging

from utils.parse_config import *
from utils.utils import build_targets, to_cpu, non_max_suppression

import matplotlib.pyplot as pyplot
import matplotlib.patches as patches

logger = logging.getLogger('main.models')


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
                    out_channels=filters,
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
        # GT与anchor iou > 0.5的位置从no_object的状态中去除
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        # 注意这里每一次使用FloatTensor的时候
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size

        # Calculate offsets for each grid, grid_x
        # 0, 1, 2, 3
        # 0, 1, 2, 3
        # 0, 1, 2, 3
        # 0, 1, 2, 3
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g,
                                                         g]).type(FloatTensor)
        # 0, 0, 0, 0
        # 1, 1, 1, 1
        # 2, 2, 2, 2
        # 3, 3, 3, 3
        self.grid_y = torch.arange(g).repeat(g,
                                             1).t().view([1, 1, g,
                                                          g]).type(FloatTensor)

        self.scaled_anchors = FloatTensor([
            (a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors
        ])
        logger.info(
            f"YOLOLayer, compute_grid_offsets: {self.scaled_anchors.size()}")

        self.anchor_w = self.scaled_anchors[:, 0:1].view(
            (1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view(
            (1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        logger.info(
            f"YOLOLayer input: {x.size(0)}, {x.size(1)}, {x.size(2)}, {x.size(3)}"
        )

        prediction = (x.view(num_samples, self.num_anchors,
                             self.num_classes + 5, grid_size,
                             grid_size).permute(0, 1, 3, 4, 2).contiguous())
        logger.info(
            f"After resize, prediction: {prediction.size(0)}, {prediction.size(1)}, {prediction.size(2)}, {prediction.size(3)}, {prediction.size(4)}"
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        pred_conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])

        # if grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_bboxes = FloatTensor(prediction[..., :4].shape)
        pred_bboxes[..., 0] = x.data + self.grid_x
        pred_bboxes[..., 1] = y.data + self.grid_y
        # 乘scale过的anchor_w, anchor_h
        pred_bboxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_bboxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat((
            pred_bboxes.view(num_samples, -1, 4) * self.stride,
            pred_conf.view(num_samples, -1, 1),
            pred_cls.view(num_samples, -1, self.num_classes),
        ), -1)
        logger.info(
            f"YOLOLayer output: {output.size(0)}, {output.size(1)}, {output.size(2)}\n"
        )

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

            # Loss : Mask outputs to ignore non-existing objects (except with conf loss)
            # 目标框使用 mse loss
            # 计算loss采用最原始的数值
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse.loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse.loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])

            # 置信度使用 bce 交叉熵, 有无物体的交叉熵比例贡献不一样
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask],
                                            tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj

            # 分类交叉熵
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])

            # 总体损失 坐标损失，置信度损失，分类损失
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            # cls_acc 不理解???
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            # detected_mask ???
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(
                iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(
                iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(
                iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss


class Darknet(nn.Module):
    """YOLOV3 object detection model"""
    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()

        # self.module_defs 词典列表，每个词典为一层的信息
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)

        # nn.Sequential()中有以下记录
        #  Sequential(
        #    (yolo_106): YOLOLayer(
        #      (mse_loss): MSELoss()
        #      (bce_loss): BCELoss()
        #    )
        #  )
        self.yolo_layers = [
            layer[0] for layer in self.module_list
            if hasattr(layer[0], "metrics")
        ]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        logger.info(f"Model input: {x.shape}")
        img_dim = x.shape[2]
        loss = 0
        # 记录每一层的输出，为route或shortcut准备
        # 返回yolo层的输出和loss
        layer_outputs, yolo_outputs = [], []
        for i, (module_def,
                module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                # 在filter上的融合
                x = torch.cat([
                    layer_outputs[int(layer_i)]
                    for layer_i in module_def["layers"].split(",")
                ], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)

        # 转到cpu上了
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        logger.info(f"Model output: {yolo_outputs.shape}\n")
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weight_path'"""
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header_info = header
            self.seen = header[3]
            weights = np.fromfile(f, dtype=np.float32)

        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(
                zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                        bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                        bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                        bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(
                        bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(
                        weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(
                    conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
        path: path of the new weights file
        cutoff: save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_into.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(
                zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load cov bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()
