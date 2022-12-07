# -*- coding= utf-8 -*-
# @Time:2022.12.07
# @Author: Jervis

import math
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
import yaml
from pathlib import Path
import cv2
from tools import check_label
from utils.check_yaml import check_yaml


def one_hot(num_class, obj_class):
    """实现numpy的one_hot编码"""
    one_hot_decode = np.zeros(num_class)
    one_hot_decode[obj_class] = 1.
    return one_hot_decode


class YOLODataset(Dataset):
    def __init__(self, yaml, is_train=True):
        self.yaml = yaml
        self.datas = []
        if is_train:
            root = Path(yaml['path'])
            train_images = root / yaml['train']
            train_labels = root / 'labels/train'
            for image in train_images.iterdir():
                image_name = image.name
                label_path = str(train_labels / image_name[:image_name.rfind('.')]) + '.txt'
                self.datas.append((image, label_path))
        else:
            root = Path(yaml['path'])
            val_images = root / yaml['val']
            val_labels = root / 'labels/val'
            for image in val_images.iterdir():
                image_name = image.name
                label_path = str(val_labels / image_name[:image_name.rfind('.')]) + '.txt'
                self.datas.append((image, label_path))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        data = self.datas[item]
        labels = {}

        image = cv2.imread(str(data[0]))
        label = check_label(data[1])

        for feature_size, anchors in self.yaml['anchors'].items():
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + self.yaml['nc']))

            for box in label:
                cls, cx, cy, w, h = box

                # 中心点偏移量
                cx_offset, cx_index = math.modf(cx * feature_size / self.yaml['input_size'][0])
                cy_offset, cy_index = math.modf(cy * feature_size / self.yaml['input_size'][1])

                for i, anchor in enumerate(anchors):
                    anchor_area = anchor[0] * anchor[1]

                    w_offset, h_offset = w / anchor[0], anchor[1]

                    area = w * h

                    iou = min(area, anchor_area) / max(area, anchor_area)

                    labels[feature_size][int(cy_index), int(cx_index), i] = np.array(
                        [iou, cx_offset, cy_offset, np.log(w_offset), np.log(h_offset), *one_hot(self.yaml['nc'], cls)]
                    )

        return labels[13], labels[26], labels[52], transforms.ToTensor()(image)


if __name__ == '__main__':
    yaml_path = r'../data/test.yaml'
    yaml = check_yaml(yaml_path)
    train_data = YOLODataset(yaml)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

    for a, b, c, image in train_loader:
        print(a.shape)
        print(b.shape)
        print(c.shape)

