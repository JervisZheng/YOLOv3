import numpy as np
from torch.utils.data import DataLoader, Dataset
import yaml
from pathlib import Path
import cv2


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


if __name__ == '__main__':
    with open(r'./coco128.yaml', encoding='ascii', errors='ignore') as file:
        yaml = yaml.safe_load(file)

    train_data = YOLODataset(yaml)
