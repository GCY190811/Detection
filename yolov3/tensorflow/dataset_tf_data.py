"""
use tf.data API
"""

import random
import tensorflow as tf
from core.config import cfg
import helper.image as img
import numpy as np
import core.utils as utils


def data_func(trainPattern, valPattern):
    train_dataset = tf.data.TFRecordDataset(
        filenames=tf.gfile.Glob(trainPattern))
    train_dataset = train_dataset.map(parse_func("train"),
                                      num_parallel_calls=6)
    # repeat 如何处理
    train_dataset = train_dataset.repeat(cfg.TRAIN.WARMUP_EPOCHS +
                                         cfg.TRAIN.FIRST_STAGE_EPOCHS +
                                         cfg.TRAIN.SECOND_STAGE_EPOCHS)

    train_dataset = train_dataset.shuffle(cfg.SHUFFLE_SIZE)
    train_dataset = train_dataset.batch(cfg.TRAIN.BATCH_SIZE).prefetch(1)

    val_dataset = tf.data.TFRecordDataset(filenames=tf.gfile.Glob(valPattern))
    val_dataset = val_dataset.map(parse_func, num_parallel_calls=6)
    # repeat 如何处理
    val_dataset = val_dataset.repeat(cfg.TRAIN.WARMUP_EPOCHS +
                                     cfg.TRAIN.FIRST_STAGE_EPOCHS +
                                     cfg.TRAIN.SECOND_STAGE_EPOCHS)
    val_dataset = val_dataset.shuffle(cfg.SHUFFLE_SIZE)
    val_dataset = val_dataset.batch(cfg.TRAIN.BATCH_SIZE).prefetch(1)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)
    val_init_op = iterator.make_initializer(val_dataset)

    return train_init_op, val_init_op, iterator.get_next()


def parse_func(mode, serialized_example):
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image':
                                           tf.FixedLenFeature([], tf.string),
                                           'bboxes':
                                           tf.FixedLenFeature([], tf.strin),
                                           'labels':
                                           tf.FixedLenFeature(dtype=tf.int64),
                                       })

    image = tf.image.decode_jpeg(features['image'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    true_boxes = tf.decode_raw(features['bboxes'], tf.float32)
    true_boxes = tf.reshape(true_boxes, shape=[-1, 4])

    true_labels = features['labels'].values

    return tf.py_func(preprocess,
                      inp=[mode, image, true_boxes, true_labels],
                      Tout=[
                          tf.uint8, tf.float32, tf.float32, tf.float32,
                          tf.float32, tf.float32, tf.float32
                      ])


def preprocess(mode, image, true_boxes, true_labels):
    input_sizes = cfg.TRAIN.INPUT_SIZE if mode == 'train' else cfg.TEST.INPUT_SIZE
    data_aug = cfg.TRAIN.DATA_AUG if mode == 'train' else cfg.TEST.DATA_AUG

    strides = np.array(cfg.YOLO.STRIDES)  # downsample ratio [8, 16, 32]
    classes = utils.read_class_names(cfg.YOLO.CLASSES)
    num_classes = len(classes)
    anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))  # numpy (3,3,2)
    anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
    max_bbox_per_scale = cfg.YOLO.MAX_BBOX_PER_SCALE

    input_size = random.choice(input_sizes)
    output_sizes = input_size // strides  # 3 output layer

    image, bboxes = image_preprocess(image, true_boxes, input_size)
    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = bboxes_preprocess(
        bboxes, true_labels, output_sizes, num_classes, anchors, strides)

    batch_image = np.zeros((input_size, input_size, 3))
    batch_label_sbbox = np.zeros(
        (output_sizes[0], output_sizes[0], anchor_per_scale, 5 + num_classes))
    batch_label_mbbox = np.zeros(
        (output_sizes[1], output_sizes[1], anchor_per_scale, 5 + num_classes))
    batch_label_lbbox = np.zeros(
        (output_sizes[2], output_sizes[2], anchor_per_scale, 5 + num_classes))
    batch_sbboxes = np.zeros((max_bbox_per_scale, 4))
    batch_mbboxes = np.zeros((max_bbox_per_scale, 4))
    batch_lbboxes = np.zeros((max_bbox_per_scale, 4))

    batch_image[:, :, :] = image
    batch_label_sbbox[:, :, :, :] = label_sbbox
    batch_label_mbbox[:, :, :, :] = label_mbbox
    batch_label_lbbox[:, :, :, :] = label_lbbox
    batch_sbboxes[:, :] = sbboxes
    batch_mbboxes[:, :] = mbboxes
    batch_lbboxes[:, :] = lbboxes

    return batch_image, batch_label_sbbox, batch_label_mbbox,
    batch_label_lbbox, batch_sbboxes, batch_mbboxes, batch_lbboxes


def image_preprocess(image, bboxes, input_size):
    """
    进行数据增强，以及图片的resize, 包含对box的处理
    水平翻转，剪裁，随机迁移
    letterbox 图片输入前，尺寸变换
    input:
        image bboxes, input_size
    return:
        image bbox
    """
    image, bboxes = img.random_horizontal_flip(np.copy(image), np.copy(bboxes))
    image, bboxes = img.random_crop(np.copy(image), np.copy(bboxes))
    image, bboxes = img.random_translate(np.copy(image), np.copy(bboxes))
    image, bboxes = img.letterBox(np.copy(image), [input_size, input_size],
                                  np.copy(bboxes))
    return image, bboxes


def bboxes_preprocess(bboxes, labels, output_sizes, num_classes, anchors,
                      strides):
    """将bboxes, labels转为计算loss时需要的格式

    Parameters
    ----------
    bboxes: numpy
        [[xmin, ymin, xmax, ymax]
         [xmin, ymin, xmax, ymax]
         ....
         ]
    labels: numpy
        (N, )
    output_sizes: list
        [input_size/8, input_size/16, input_size/32]
    num_classes: int
    anchors: numpy
        (3, 3, 2)

    Returns
    -------
        计算loss中使用的bbox

    """
    print("bboxes".format(bboxes))
    print("labels".format(labels))

    bboxes_label = [
        np.zeros((output_sizes[i], output_sizes[i], cfg.YOLO.ANCHOR_PER_SCALE,
                  5 + num_classes)) for i in range(3)
    ]
    bboxes_xywh = [
        np.zeros((cfg.YOLO.MAX_BBOX_PER_SCALE, 4)) for _ in range(3)
    ]
    bboxes_count = np.zeros((3, ))

    # 存在有效的box
    exist_positive = False
    iou_statistic = []
    for bbox_index in range(len(bboxes)):
        bboxCoor = bboxes[bbox_index]
        bboxClassIndex = labels[bbox_index]

        # smooth label
        onehot = np.zeros(num_classes, dtype=np.float)
        onehot[bboxClassIndex] = 1.0
        uniform_distribution = np.full(num_classes, 1.0 / num_classes)
        deta = 0.01
        smoothOnehot = onehot * (1 - deta) + deta * uniform_distribution

        # bbox [xcernter, ycenter, width, height]
        bbox_xywh = np.concatenate(
            [(bboxCoor[[0, 1]] + bboxCoor[[2, 3]]) * 0.5,
             (bboxCoor[2:] - bboxCoor[:2])],
            axis=-1)

        for i in range(len(output_sizes)):  # 不同的scale
            # GT的中心点，anchor的长和高
            bbox_anchor = np.zeros((cfg.YOLO.ANCHOR_PER_SCALE, 5))
            bbox_anchor[:, :2] = (bbox_xywh / strides[i])[:2]
            bbox_anchor[:, 2:4] = anchors[i]

            # 一种尺度下，一个gtbox和多种anchor的交并比
            iou_score = img.iou(bbox_xywh, bbox_anchor, method="xywh")
            # > 0.3认为这个box属于这个尺度下的，这个anchor
            iou_threshold = iou_score > 0.3
            iou_statistic.append(iou_score)

            if np.any(iou_threshold):
                # 找到每个box的位置
                indx, indy = np.floor(bbox_anchor[:, :2]).astype(np.int32)
                # 先清零, 使用iou_threshold这中骚操作
                bboxes_label[i][indy, indx, iou_threshold, :] = 0
                bboxes_label[i][indy, indx, iou_threshold, :4] = bbox_xywh
                bboxes_label[i][indy, indx, iou_threshold, 4] = 1
                bboxes_label[i][indy, indx, iou_threshold, 5:] = smoothOnehot

                # bboxes找位置
                boxIndex = bboxes_count[i] % cfg.YOLO.MAX_BBOX_PER_SCALE
                bboxes_xywh[i][boxIndex, :4] = bbox_xywh
                bboxes_count[i] += 1

                exist_positive = True

        if not exist_positive:
            maxIndex = np.argmax(np.array(iou_statistic).reshape(-1))
            best_scale = int(maxIndex / cfg.YOLO.ANCHOR_PER_SCALE)
            best_anchor = int(maxIndex % cfg.YOLO.ANCHOR_PER_SCALE)
            xind, yind = np.floor(
                (bbox_xywh / strides[best_scale])[:2]).astype(np.int32)

            bboxes_label[best_scale][yind, xind, best_anchor, :] = 0
            bboxes_label[best_scale][yind, xind, best_anchor,
                                     0:4] = bboxes_xywh
            bboxes_label[best_scale][yind, xind, best_anchor, 4] = 1.0
            bboxes_label[best_scale][yind, xind, best_anchor,
                                     5:] = smoothOnehot

            boxIndex = bboxes_count[best_scale] % cfg.YOLO.MAX_BBOX_PER_SCALE
            bboxes_xywh[best_scale][boxIndex, :4] = bboxes_xywh
            bboxes_count[best_scale] += 1

    # 注意这里的 l,m,s 定义与原始代码作者的含义不一样, 返回值也不一样
    label_lbbox, label_mbbox, label_sbbox = bboxes_label
    lbboxes, mbboxes, sbboxes = bboxes_xywh
    return label_lbbox, label_mbbox, label_sbbox, lbboxes, mbboxes, sbboxes
