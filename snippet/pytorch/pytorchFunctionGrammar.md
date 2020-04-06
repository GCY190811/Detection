## Grammar  
-----  
> - torch.mm()  
数学上的矩阵乘法，没有broadcast  

> -  torch.clamp(input, min, max, out=None)  
将变量限定在一定范围  

> - torch.pow(input, exponent, out=None)  
input, exponent都可以是tensor, 存在broadcast, 结果尺寸由两者决定  

> - item()
当tensor纬度是1, 返回tensor的数值， 标量， 此操作不可求导  

> - torch.t(input)  
2-D tensor的转置， 可以认为是 transpose(input, 0, 1)  

> - clone()  
拷贝tensor，需要注意梯度传播对两个变量的影响

> - torch.no_grad()  
背景管理器， 所有在此环境下计算的结果，均不会放入自动图中进行梯度更新 

> - grad  
ex: w1.grad  
默认是None, 当调用了backward()对变量进行求导后, .grad成为tensor保存梯度。再次调用backward()时，默认是累加操作。  

> - zero_()  
将这个tensor填满0。 常用于在计算梯度后，清零梯度。  

### nn module  
-----
nn module: pytorch high-level package like Keras, TensorFlow-Slim, TFLearn.  
用户只需管理输入即可
> - torch.nn.Sequential()  
指定尺寸变化，建立模型函数，按照输入顺序执行。可以：  
model = torch.nn.Sequential(  
&ensp;&ensp;&ensp;&ensp;     torch.nn.Linear(D_in, H),  
&ensp;&ensp;&ensp;&ensp;     torch.nn.ReLU(),  
&ensp;&ensp;&ensp;&ensp;     torch.nn.Linear(H, D_out),  
    )  
y_pred = model(x)  
model.zero_grad()  
for param in model.parameters():

> - torch.nn.Linear()  

> - torch.nn.ReLU()  

> - torch.nn.MSELoss(reduction='sum')  
返回计算损失的函数，并且可以对维度进行缩减  
loss_fn = torch.nn.MSELoss(reduction='sum')
loss = loss_fn(y_pred, y)  

### torch.optim  
---
各种优化算法
> - optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
optimizer.zero_grad() 代替 model.zero_grad()  
optimizer.step() 代替：  
with torch.no_grad():  
&ensp;&ensp;&ensp;&ensp;    for param in model.parameters():  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;        param -= params.grad * learning_rate

### custom nn modules
----
> - class TwoLayerNet(torch.nn.Module):  
&ensp;&ensp;&ensp;&ensp; def \_\_init__(self, TwoLayerNet):  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; super(TwoLayerNet, self).\_\_init__()  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; self.linear1 = torch.nn.Linear(D_in, H)  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; self.linear2 = torch.nn.Linear(H, D_out)  
&ensp;&ensp;&ensp;&ensp;  def forward(self, x):  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; h_relu = self.linear1(x).clamp(min=0)  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; y_pred = self.linear2(h_relue)  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp; retrun y_pred 