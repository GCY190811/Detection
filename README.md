# Detection

## YOLOv3

### model(inference) size

Only save model trainable parameters, torch.save(model.state_dict())

**params(counts)**
*Conv2d*: Cin×Cout×K×K
*BatchNorm2d*: 2N (N is conv filters)
*LeakyReLU*: 0
*Upsample*: 0

**MACs & FLOPs**

```markdown
FLOPs is abbreviation of floating operations which includes mul / add / div ... etc.
MACs stands for multiply–accumulate operation that performs a <- a + (b x c).
As shown in the text, one MACs has one mul and one add. That is why in many places FLOPs is nearly two times as MACs.

When comparing MACs /FLOPs, we want the number to be implementation-agnostic and as general as possible. Therefore in THOP, we only consider the number of multiplications and ignore all other operations.

PS: The FLOPs is approximated by multiplying two.
```

原始代码中 BatchNorm2d 尺寸为 2N, 从保存的模型来看，应为 4N, 分别是 (weight, bias, running_mean, running_var)

```markdown
yolov3, input(3, 416, 416)
================================================================
Total params: 61,949,149
Trainable params: 61,949,149
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 1.98
Forward/backward pass size (MB): 901.03
Params size (MB): 236.32
Estimated Total Size (MB): 1139.33
----------------------------------------------------------------
```

### 模型理解

网络结构

*shortcut*:  
相应层直接叠加  

*route*:  
相应层进行cat操作，在filter的维度上进行叠加  

*num_classes*:  
模型最后一个 conv，filter 数量由 anchor_num*(num_class+5) 决定。tensorflow 由输入参数部分修改，pytorch 由 config 中的 crate_custom_model.sh 决定。  

*output*:  
x,y: (sigmoid(x/y) + grid_x/y)*stride.  
x,y在sigmoid后，位于[0,1]之间，grid_size由输出的feature_map中的维度决定。grid_x/y其实就是[0,1,2，...13),  
[0,1,2...26), [0,1,2...52). 原图中的网格映射到feature map中的一个点。只需判断，x/y偏移了多少，  
就是在对应的网格中进行了相应的偏移。当x/y为0.5时，此时预测的框中心与anchor重合。  

w/h: exp(w/h) * anchor_w/h * stride.  
这里anchor_w/h 是 anchor_w/h / stride后的。  
当w/h位于0时，exp(w/h)位于1，此时预测的w,h结果正好与anchor的尺寸一致。  

pred_conf/pred_cls: sigmoid(pred_conf/pred_cls)统一在(0, 1)之间.  

*loss*  

*metrics*  

*训练流程*  



## Todo

__pytorch-yolov3__

- [ ] 修改 torchsummary, BatchNorm2d 参数量为 4×N
- [x] 模型 flops 统计
- [ ] mixup
- [x] test  
- [ ] evaluate  
- [ ] train

__pytorch-ssd__

- [ ] SSD
- [x] test  
- [ ] evaluate
- [ ] train

__pytorch-anchor-free__

## reference

<https://github.com/eriklindernoren/PyTorch-YOLOv3>  
<https://github.com/sksq96/pytorch-summary>  
<https://github.com/Lyken17/pytorch-OpCounter>  

<https://github.com/qfgaohao/pytorch-ssd>  
<https://zhuanlan.zhihu.com/p/33544892>  
