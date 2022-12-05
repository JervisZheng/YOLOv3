from iou import cal_iou
import numpy as np
import cv2


def nms(boxes, thresh):
    if boxes.shape[0] == 0:
        return np.array([])

    _boxes = boxes[(-boxes[:, 0]).argsort()]
    r_boxes = []

    while _boxes.shape[0] > 1:
        a_box = _boxes[0]
        b_boxes = _boxes[1:]

        r_boxes.append(a_box)

        index = np.where(cal_iou(a_box, b_boxes) < thresh)
        _boxes = b_boxes[index]
    if _boxes.shape[0] > 0:
        r_boxes.append(_boxes[0])

    return np.stack(r_boxes)

