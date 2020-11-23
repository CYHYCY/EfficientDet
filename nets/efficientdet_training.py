from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import cv2
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from PIL import Image


def preprocess_input(image):
    image /= 255
    mean = (0.406, 0.456, 0.485)
    std = (0.225, 0.224, 0.229)
    image -= mean
    image /= std
    return image


def calc_iou(a, b):
    """
    a: shape is (H*W*9+……, 4)
    b: shape is (M, 4)
    return:
    IoU shape is (H*W*9+……, M)
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua

    return IoU


def get_target(anchor, bbox_annotation, classification, cuda):
    """
    params:
        anchor: shape is (1, H*W*9+……, 4), predict anchor
        bbox_annotation: shape is (M, 5), M is Number of labels
        classification: shape is (H*W*9+……, num_classes)
    return:
        targets: shape is (H*W*9+……, num_classes), value is 0 or 1 or -1,
        num_positive_anchors: Number of positive samples
        positive_indices: shape is H*W*9+……, Index of positive sample
        assigned_annotations: shape is (H*W*9+……, 5), bbox_annotation corresponding to the IoU_max
    """
    IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])  # shape is (H*W*9+……, M)

    # 计算每个框到底属于哪个label的，这样操作，会出现每个先验框只能属于其中一种框这种情况
    # IoU_argmax表示先验框anchor和哪个bbox_annotation重叠度高所对应的索引
    IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # shape is H*W*9+……

    # compute the loss for classification，每个框属于哪一类别，先初始化这个targets
    targets = torch.ones_like(classification) * -1  # shape is (H*W*9+……, num_classes), 初始值为-1
    if cuda:
        targets = targets.cuda()

    targets[torch.lt(IoU_max, 0.4), :] = 0  # IoU小于0.4返回True，会智能广播，IoU_max值小于0.4的这一行全部设置为0，即背景

    positive_indices = torch.ge(IoU_max, 0.5)  # IoU大于0.5返回True，会智能广播，shape is H*W*9+……, 存储正例样本索引

    num_positive_anchors = positive_indices.sum()  # 正例样本数

    # 对应位置存放bbox_annotation的信息。存放IoU最大的bbox_annotation的框的信息，shape is (H*W*9+……, 5)
    assigned_annotations = bbox_annotation[IoU_argmax, :]  # 根据IoU_argmax里的值作为索引,扩充bbox_annotation

    targets[positive_indices, :] = 0  # 先将正例样本所对应的那一行全部设置为0，之后再在对应位置上的那一列设置为1
    # assigned_annotations[positive_indices, 4]表示正例样本的类别信息
    targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
    return targets, num_positive_anchors, positive_indices, assigned_annotations


def encode_bbox(assigned_annotations, positive_indices, anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y):
    """
    params:
        assigned_annotations: shape is (H*W*9+……, 5), bbox_annotation corresponding to the IoU_max，对应标签的信息
        positive_indices: shape is H*W*9+……, Index of positive sample
        anchor_widths: 框(自动生成的)的宽度, shape is H*W*9+……
        anchor_heights: 框(自动生成的)的高度
        anchor_ctr_x: 框中心坐标x轴信息
        anchor_ctr_y: 框中心坐标y轴信息
    return:
        targets: shape is (M, 4), targets_dy, targets_dx, targets_dh, targets_dw
    """
    assigned_annotations = assigned_annotations[positive_indices, :]  # 对应有物体的anchor，shape is (M, 5)

    anchor_widths_pi = anchor_widths[positive_indices]  # shape is M
    anchor_heights_pi = anchor_heights[positive_indices]  # shape is M
    anchor_ctr_x_pi = anchor_ctr_x[positive_indices]  # shape is M
    anchor_ctr_y_pi = anchor_ctr_y[positive_indices]  # shape is M

    # groundtruth，由assigned_annotations计算
    gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]  # shape is M
    gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]  # shape is M
    gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths  # shape is M，中心坐标x轴信息
    gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights  # shape is M

    # efficientdet style
    gt_widths = torch.clamp(gt_widths, min=1)
    gt_heights = torch.clamp(gt_heights, min=1)

    targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi  # (标签x轴值-生成框x轴值)/生成框的宽
    targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi  # (标签x轴值-生成框y轴值)/生成框的高
    targets_dw = torch.log(gt_widths / anchor_widths_pi)  # log(标签宽/生成框的宽)
    targets_dh = torch.log(gt_heights / anchor_heights_pi)  # log(标签高/生成框的高)

    targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw))  # shape is (M, 4)
    targets = targets.t()  # shape is (M, 4)
    return targets


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, classifications, regressions, anchors, annotations, alpha=0.25, gamma=2.0, cuda=True):
        """
        classifications: shape is (batch_size, H*W*9+……, num_classes), predict
        regressions: shape is (batch_size, H*W*9+……, 4) predict
        anchors: shape is (1, H*W*9+……, 4) 根据网格生成的框的信息，之后还需结合regressions信息生成真正的预测框
        annotations: type is list len(annotations)==batch_size, annotations[0]存储坐标信息和类别信息

        """
        # 设置
        dtype = regressions.dtype
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []  # len(regression_losses) == batch_size, 每个元素里面只有一个数

        # 获得先验框，将先验框转换成中心宽高的形势
        anchor = anchors[0, :, :].to(dtype)
        # 转换成中心，宽高的形式
        anchor_widths = anchor[:, 3] - anchor[:, 1]  # shape is H*W*9+……
        anchor_heights = anchor[:, 2] - anchor[:, 0]  # shape is H*W*9+……
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths  # shape is H*W*9+……
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights  # shape is H*W*9+……

        for j in range(batch_size):
            # 取出真实框
            bbox_annotation = annotations[j]

            # 获得每张图片的分类结果和回归预测结果
            classification = classifications[j, :, :]  # shape is (H*W*9+……, num_classes)
            regression = regressions[j, :, :]  # shape is (H*W*9+……, 4), targets_dy, targets_dx, targets_dh, targets_dw
            # 平滑标签
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if len(bbox_annotation) == 0:
                alpha_factor = torch.ones_like(classification) * alpha

                if cuda:
                    alpha_factor = alpha_factor.cuda()
                alpha_factor = 1. - alpha_factor
                focal_weight = classification
                focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                bce = -(torch.log(1.0 - classification))

                cls_loss = focal_weight * bce

                if cuda:
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                else:
                    regression_losses.append(torch.tensor(0).to(dtype))
                classification_losses.append(cls_loss.sum())
                continue

            # 获得目标预测结果
            # targets: shape is (H*W*9+……, num_classes), value is 0 or 1 or -1
            # num_positive_anchors: Number of positive samples
            # positive_indices: shape is H*W*9+……, Index of positive sample
            # assigned_annotations: shape is (H*W*9+……, 5), bbox_annotation corresponding to the IoU_max, 标签信息
            targets, num_positive_anchors, positive_indices, assigned_annotations = get_target(anchor, bbox_annotation,
                                                                                               classification, cuda)

            alpha_factor = torch.ones_like(targets) * alpha  # shape is (H*W*9+……, num_classes)
            if cuda:
                alpha_factor = alpha_factor.cuda()
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)  # 1-y^hat
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)  # alpha*(1-y^hat)^gamma
            # focal_weight shape is (H*W*9+……, num_classes)
            # CE
            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce  # focal loss

            zeros = torch.zeros_like(cls_loss)
            if cuda:
                zeros = zeros.cuda()
            # 只计算标签为0和1的损失
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)  # torch.ne表示不等于就返回True
            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))
            # smoooth_l1，计算回归损失
            if positive_indices.sum() > 0:  # 正例的框的个数
                targets = encode_bbox(assigned_annotations, positive_indices, anchor_widths, anchor_heights,
                                      anchor_ctr_x, anchor_ctr_y)
                # shape is (M, 4)
                regression_diff = torch.abs(targets - regression[positive_indices, :])  # 欧氏距离
                # le小于为True
                regression_loss = torch.where(torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2), regression_diff - 0.5 / 9.0)
                regression_losses.append(regression_loss.mean())
            else:
                if cuda:
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                else:
                    regression_losses.append(torch.tensor(0).to(dtype))

        c_loss = torch.stack(classification_losses).mean()  # 只有一个值
        r_loss = torch.stack(regression_losses).mean()
        loss = c_loss + r_loss
        return loss, c_loss, r_loss


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


class Generator(object):
    def __init__(self, batch_size,
                 train_lines, image_size,
                 ):

        self.batch_size = batch_size
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5):
        '''r实时数据增强的随机预处理'''
        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        # resize image
        new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
        scale = rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # place image
        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        # correct boxes
        box_data = np.zeros((len(box), 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box
        if len(box) == 0:
            return image_data, []

        if (box_data[:, :4] > 0).any():
            return image_data, box_data
        else:
            return image_data, []

    def generate(self):
        while True:
            shuffle(self.train_lines)
            lines = self.train_lines
            inputs = []
            targets = []
            n = len(lines)
            for i in range(len(lines)):
                img, y = self.get_random_data(lines[i], self.image_size[0:2])
                i = (i + 1) % n
                if len(y) != 0:
                    boxes = np.array(y[:, :4], dtype=np.float32)
                    y = np.concatenate([boxes, y[:, -1:]], axis=-1)

                img = np.array(img, dtype=np.float32)
                y = np.array(y, dtype=np.float32)
                inputs.append(np.transpose(preprocess_input(img), (2, 0, 1)))
                targets.append(y)
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    yield tmp_inp, tmp_targets
