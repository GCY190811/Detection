import torch
import torch.nn as nn
from torch.autograd import Variable

input = torch.ones(1, 3, 224, 224)
input = Variable(input)
f = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=5, groups=3)
output = f(input)
print(output.shape)
