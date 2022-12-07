from pathlib import Path
import cv2
import numpy as np
from utils.check_yaml import check_yaml
import xml.etree.cElementTree as ET
import yaml


def diag2cent(box):
    """对角坐标转化为中心点坐标"""
    x = box[0] + (box[2] - box[0]) // 2
    y = box[1] + (box[3] - box[1]) // 2
    w = box[2] - box[0]
    h = box[3] - box[1]
    return np.array([x, y, w, h])


def voc2yolo(image, boxes, outsize):
    """voc格式数据，转化为yolo格式"""
    h, w, _ = image.shape
    new_image = np.zeros((outsize, outsize, 3), dtype=np.uint8)

    ratio = outsize / max(h, w)  # 等比缩放比例

    image_ = cv2.resize(image, (int(w * ratio), int(h * ratio)))

    new_h, new_w, _ = image_.shape
    (y, x) = ((outsize - new_h) // 2, (outsize - new_w) // 2)

    new_image[y:y + new_h, x:x + new_w, :] = image_

    boxes *= ratio
    boxes += np.array([x, y, x, y])
    boxes = diag2cent(boxes)
    # boxes = boxes / outsize
    return new_image, boxes


def voc_parse(xml_path, config_path):
    """
    xml文件解析， 图片后缀为jpg
    """
    root = Path(__file__).parents[1]
    xml_path = Path(xml_path)
    xml_dir = xml_path / 'Annotations'
    labels_dir = root / 'dataset' / 'labels'
    images_dir = root / 'dataset' / 'images'

    cfg = check_yaml(config_path)

    if not labels_dir.exists():
        labels_dir.mkdir(parents=True)

    if not images_dir.exists():
        images_dir.mkdir(parents=True)

    if xml_dir.is_dir():
        for xml_file in xml_dir.iterdir():
            image_name = xml_file.name[:xml_file.name.rfind('.')] + '.jpg'
            image_path = xml_path / 'images' / image_name

            label_name = xml_file.name[:xml_file.name.rfind('.')] + '.txt'

            label_save_path = labels_dir / label_name
            image_save_path = images_dir / image_name

            try:
                image = cv2.imread(str(image_path))
            except IOError:
                print('image not find')

            tree = ET.parse(str(xml_file))
            root = tree.getroot()

            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls not in cfg['nc']:
                    continue

                cls_id = cfg['nc'].index(cls)
                bndbox = obj.find('bndbox')

                xmin = bndbox.find('xmin').text
                ymin = bndbox.find('ymin').text
                xmax = bndbox.find('xmax').text
                ymax = bndbox.find('ymax').text

                box = np.array([cls_id, xmin, ymin, xmax, ymax]).astype(np.float32)
                new_image, box[1:5] = voc2yolo(image, box[1:5], 640)

                with open(label_save_path, 'a+', encoding='utf-8') as file:
                    file.write(" ".join([str(x) for x in box]) + '\n')

            cv2.imwrite(str(image_save_path), new_image)


def visualization(origin_image, box, insize):
    pass


def check_label(label_path):
    label_path = Path(label_path)
    if label_path.suffix == '.txt':
        with open(str(label_path), 'r', encoding='utf-8') as file:
            objs = file.read().splitlines()

        boxes = []
        for obj in objs:
            box = obj.strip(' ').split(' ')
            box = list(map(int, list(map(float, box))))
            boxes.append(box)

        return np.array(boxes)


if __name__ == '__main__':
    # path = r'origindata'
    # yaml_dir = r'../models/yolov3.yaml'
    # voc_parse(path, yaml_dir)

    label_path = r'../dataset/labels/train/5.txt'
    boxes = check_label(label_path)
    print(boxes)
