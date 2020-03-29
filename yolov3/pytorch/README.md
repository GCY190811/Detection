# 记录

## Detect

### Darknet 构造网络

解析模型的 cfg 文件 => 得到 module_defs.

搭建模型 (create_modules) => 模型的参数输入情况；使用 nn.ModuleList() 搭建的 module_list.

从 module_list 中提取出 yolo_layer = >

```shell
[YOLOLayer(
  (mse_loss): MSELoss()
  (bce_loss): BCELoss()
), YOLOLayer(
  (mse_loss): MSELoss()
  (bce_loss): BCELoss()
), YOLOLayer(
  (mse_loss): MSELoss()
  (bce_loss): BCELoss()
)]
```

### 载入模型

```python
model.load_stat_dict(torch.load(weights_path))
```

### 切换为评估模式

```python
model.eval()
```

### 准备数据

DataLoader + 自定义数据类 => dataloader 对象

自定义类对象：迭代器__getitem__方法，取出图片，将图片 Pading 成正方形，并 resize 到指定尺寸。

加载类别文件 => load_classes

### inference

在 with torch.no_grad() 作用域中：输入图片，model 进行 forward 操作 => 得到 detections 结果。

Darknet::forward

```python
    def forward(self, x, targets=None):
        logger.info(f"input_img shape: {x.shape}")
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

        logger.info(f"yolo_layer output1 : {yolo_outputs[0].shape}")
        logger.info(f"yolo_layer output2 : {yolo_outputs[1].shape}")
        logger.info(f"yolo_layer output3 : {yolo_outputs[2].shape}")
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs
```

记录每一层输出，为 route 或 shortcut 做准备。
三层 yolo_layer 输出的结果 cat(yolo_outputs, 1) 后返回，此时没有输入 target, 即没有 loss 的输出。

YOLOLayer::forward

现有的网络结构将输入的图片放缩了 32, 16, 8 倍 (stride)，若输入为 416x416， 则 feature map 为 13x13, 26x26, 52x52 的 (grid_size) 尺寸

举例：将 input(1, 255,13, 13) view => (1, 3, 13, 13, 85)

```python
prediction = (x.view(num_samples, self.num_anchors,
                             self.num_classes + 5, grid_size,
                             grid_size).permute(0, 1, 3, 4, 2).contiguous())
```

YOLOLayer::compute_grid_offsets

self.stride => 32. 在原图相对缩放了 32 倍，feature map(13x13) 中的每个点相当于原图中的 32x32 图像块。

self.grid_x, grid_y => 相当于是 feature map 中的 index 矩阵。

self.scaled_anchor => anchor/stride, anchor 中最大的三个对应于第一个 yolo layer，因为此时的缩放比例最大，stride=32.

self.anchor_w, self.anchor_h => scale_anchor 分开，尺寸为 (1, 3, 1, 1)

```python
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
        logger.info(f"YOLOLayer, compute_grid_offsets: {self.scaled_anchors.size()}")

        self.anchor_w = self.scaled_anchors[:, 0:1].view(
            (1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view(
            (1, self.num_anchors, 1, 1))
```

Darknet::forward

prediction[..., 0] => (sigmoid(x) + self.grid_x) * stride

prediction[..., 1] => (sigmoid(y) + self.grid_y) * stride

prediction[..., 2] => exp(w) * self.anchor_w * stride

prediction[..., 3] => exp(h) * self.anchor_h * stride

prediction[..., 4] => sigmoid(confidence)

prediction[..., 5:] => sigmoid(class_score)

返回时将检测结果压缩到三维 => (num_samples, anchor_num*feature_map_size^2, class_num+5)
此时没有 target 部分直接返回 output 结果。

```python
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
```

将三层检测结果 cat => (num_samples,  + feature_map_size^feature_map_size*anchor for layer in yolo_layers, num_class+5)

### non_max_suppression

utils::non_max_suppression

排除 conf 小于 conf_thres 的目标框 => 干掉大量检测结果

```python
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        logger.info(f"NonMaxSuppression after conf: {image_pred.size()}")
```

检测结果排序，依据 confidence*和预测的类别概率 => 检测结果简化到 (n, 7), n: 目标框数量，7: xmin,ymin,xmax,ymax,cofidence,class_confidence,class_index

```python
        # Object confidence times class confidence
        # pytorch中的max()后，返回的第一维是最大值，第二位返回对应的index
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]

        # Sort by it
        # argsort()返回升序排列的Index列表，这里使用-score，相当于score按照降序排列的index
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat(
            (image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        logger.info(f"NonMaxSuppression after sort: {detections.size()}")
```

找到置信度最大的一个框，查找 iou 重叠率大于阈值并且预测的类别相同的框，*将这些框按照置信度进行合并，找到一个融合的新框作为这个检测的结果*。将这些框从序列中排除。循环以上步骤，直到没有侯选框为止。最后将所有结果框 stack 一下，作为这张图片的检测结果。

```python
        keep_boxes = []
        while detections.size(0):
            # unsqueeze在0维度处添加一维度
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0),
                                     detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (
                weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)
```  

