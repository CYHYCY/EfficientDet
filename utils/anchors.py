import itertools
import torch
import torch.nn as nn
import numpy as np


class Anchors(nn.Module):
    """
    5中feature map上的cell生成先验框，即左上右下坐标
    """
    def __init__(self, anchor_scale=4., pyramid_levels=None):
        super().__init__()
        self.anchor_scale = anchor_scale

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]

        self.strides = [2 ** x for x in self.pyramid_levels]  # 原始图片下采样的倍数[8, 16, 32, 64, 128]
        # 一个scales对应3个ratios,所以有一个点对应9个先验框
        self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])  # [1., 1.259, 1.587]
        self.ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]

    def forward(self, image):
        """
        image: shape is (batch_size, 3, H, W), 真实图片的高宽
        return:
        anchor_boxes: shape is (1, H*W*9+……, 4), (y1, x1, y2, x2)
        例如，input_size=768, 此时shape is (1, (96^2+48^2+24^2+12^2+6^2)*9, 4)=(1, 110484, 4)

        """
        image_shape = image.shape[2:]

        boxes_all = []
        for stride in self.strides:
            boxes_level = []
            for scale, ratio in itertools.product(self.scales, self.ratios):
                if image_shape[1] % stride != 0:
                    raise ValueError('input size must be divided by the stride.')
                base_anchor_size = self.anchor_scale * stride * scale  # 基础框的尺寸
                anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0  # 候选框宽度的一半
                anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0  # 候选框高度的一半
                x = np.arange(stride / 2, image_shape[1], stride)
                y = np.arange(stride / 2, image_shape[0], stride)

                xv, yv = np.meshgrid(x, y)  # 中心坐标的值，每一种feature map的中心位置坐标不变，变的是高宽信息

                xv = xv.reshape(-1)
                yv = yv.reshape(-1)

                # y1,x1,y2,x2
                boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                                   yv + anchor_size_y_2, xv + anchor_size_x_2))  # shape is (4, -1)
                boxes = np.swapaxes(boxes, 0, 1)  # shape is (-1, 4)
                # boxes_level is list, len(boxes_level)==9, boxes_level[0] shape is (-1, 1, 4)
                boxes_level.append(np.expand_dims(boxes, axis=1))
            # concat anchors on the same level to the reshape NxAx4
            boxes_level = np.concatenate(boxes_level, axis=1)  # shape is (24**2, 9, 4) or (48**2, 9, 4)
            # boxes_all表示对应feature map上候选框的汇总
            boxes_all.append(boxes_level.reshape([-1, 4]))  # len(boxes_all) == 5 boxes_all[0] shape is (-1, 4)

        anchor_boxes = np.vstack(boxes_all)  # shape is (-1, 4), 5中feature map上候选框的总和

        anchor_boxes = torch.from_numpy(anchor_boxes).to(image.device)  # shape is (-1, 4)
        anchor_boxes = anchor_boxes.unsqueeze(0)  # shape is (1, H*W*9+……, 4), 4表示左上右下的坐标信息

        return anchor_boxes
