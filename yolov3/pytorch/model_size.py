""" save model only containing parameters"""

from models import *
from torchsummary import summary

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def",
                        type=str,
                        default="config/yolov3.cfg",
                        help="path to model definition file")
    parser.add_argument(
        "--weights_path",
        type=str,
        default=
        "/home/guo/myDisk/DatasetFiles/coco/model/checkpoints/yolov3_ckpt_10.pth",
        help="path to weights file")
    parser.add_argument(
        "--target_model",
        type=str,
        default="/home/guo/Models/yolo/target_model",
        help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def).to(device)

    # 统计信息
    summary(model, input_size=(3, 416, 416))

    # 加载模型，查看模型中的参数量
    state_dict = torch.load(opt.weights_path)
    model.load_state_dict(state_dict)

    # show keys
    for k, v in state_dict.items():
        print(k, "\t", state_dict[k].size())
    
    

