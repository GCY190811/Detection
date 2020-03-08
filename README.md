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



原始代码中BatchNorm2d尺寸为2N, 从保存的模型来看，应为4N, 分别是(weight, bias, running_mean, running_var)

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

## Todo  

- [ ] 修改torchsummary, BatchNorm2d参数量为4×N  
- [x] 模型flops统计
- [ ] SSD

## reference  

<https://github.com/eriklindernoren/PyTorch-YOLOv3>  
<https://github.com/sksq96/pytorch-summary>  
<https://github.com/Lyken17/pytorch-OpCounter>  
