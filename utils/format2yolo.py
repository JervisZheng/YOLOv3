# -*- coding: UTF-8 -*-
# @Time: 2022.12.05
# @author: Jervis
import cv2
import numpy as np
import yaml
from pathlib import Path


def diag2cent(box, image_size):
    x = box[0] + (box[2] - box[0]) // 2
    y = box[1] + (box[3] - box[1]) // 2
    w = box[2] - box[0]
    h = box[3] - box[1]
    return x, y, w, h


def


def voc2yolo1(images_path, image_outsize, save_path, label_path=None):
    """
    图片进行填充, 训练阶段有标签，标签作相应变换
    """
    save_path = Path(save_path)
    images_path = Path(images_path)
    if not save_path.exists():
        save_path.mkdir()
    image = cv2.imread(str(images_path))
    image_name = images_path.name
    label_name = image_name[:image_name.rfind('.')] + '.txt'
    label_path = save_path / 'labels'

    h, w, _ = image.shape
    new_images = np.zeros((image_outsize, image_outsize, 3), dtype=np.uint8)

    ratio = image_outsize / max(h, w)  # 等比缩放比例

    image_ = cv2.resize(image, (int(w * ratio), int(h * ratio)))

    new_h, new_w, _ = image_.shape
    (y, x) = ((image_outsize - new_h) // 2, (image_outsize - new_w) // 2)

    new_images[y:y + new_h, x:x + new_w, :] = image_
    cv2.imwrite(new_images, save_path)

    if label_path:
        # 坐标转换
        with open(label_path, 'r', encoding='utf-8') as file:
            obj_boxes = file.read().splitlines()

        boxes = []

        for obj in obj_boxes:
            obj_boxes = np.array(list(map(float, obj.split())))
            obj_boxes[1:] = obj_boxes[1:] * ratio
            obj_boxes[1:5] = obj_boxes[1:5] + np.array([x, y, x, y])  # 计算填充后的中心点坐标
            boxes.append(obj_boxes)

        boxes = np.array(boxes, dtype=np.int32)
        boxes[1:5] = diag2cent(boxes[1:5])
        for box in boxes:
            with open(save_path, 'a+', encoding='utf-8') as file:
                file.write(" ".join([str(x) for x in box]))


if __name__ == '__main__':
    image_path = r'./001.jpg'
    label_path = r'./label.txt'
    image, boxes = voc2yolo(image_path, 640, label_path)
    for box in boxes:
        image = cv2.rectangle(image, (box[1], box[2]), (box[3], box[4]), (0, 0, 255), 1)
    cv2.imshow('1', image)
    cv2.waitKey()
