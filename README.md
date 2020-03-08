# Detection  

## YOLOv3  

### model(inference) size  
Only save model trainable parameters, torch.save(model.state_dict())  

**params(counts)**  
*Conv2d*: Cin×Cout×K×K  
*BatchNorm2d*: 2N (N is conv filters)  
*LeakyReLU*: 0  
*Upsample*: 0

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

## Todo:  

- [ ] 修改torchsummary, BatchNorm2d参数量为4×N  
- [ ] 模型flops统计
- [ ] SSD

## reference  

<https://github.com/eriklindernoren/PyTorch-YOLOv3>  
<https://github.com/sksq96/pytorch-summary>  
