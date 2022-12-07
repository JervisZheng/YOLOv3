import numpy as np


def or_iou(box, boxes):
    box_area = (box[3] - box[1]) * (box[2] - box[0])
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2], box[:, 0])

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.maximum(box[2], boxes[:, 2])
    y2 = np.maximum(box[3], boxes[:, 3])

    w = np.maximum(x2 - x1, 0)
    h = np.maximum(y2 - y1, 0)

    inter = w * h

    iou = np.divide(inter, box_area + area - inter)

    return iou


