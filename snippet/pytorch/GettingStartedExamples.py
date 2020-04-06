import numpy as np
import torch

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)     #torch.rand()定义的变量是tensor
y = torch.randn(N, D_out)

model = TwoLayerNet(D_in, H, D_out)

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
    y_pred = model(x)

    loss = criterion(y_pred, y)
    print(t, loss.item())  #标量

    # model.zero_grad()
    optimizer.zero_grad()    #optim模块接管了梯度计算部分
    loss.backward()
    optimizer.step()    #进行参数的梯度更新
    # with torch.no_grad():   #在自动图中不更新,否则梯度下降的过程也需要反向梯度了
    #     for param in model.parameters():
    #         param -= param.grad * learning_rate




