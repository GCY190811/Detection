import torch
import torchvision.models as models

resnet = torch.jit.trace(models.resnet18(), torch.rand(1, 3, 224, 224))
output = resnet(torch.ones(1, 3, 224, 224))
print(output)
resnet.save('resnet.pt')