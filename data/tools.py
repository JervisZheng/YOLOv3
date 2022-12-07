from pathlib import Path
import numpy as np
import xml.etree.cElementTree as ET
import yaml


def voc_parse(path, yaml_file):
    """
    xml文件解析
    """
    path = Path(path)
    xml_dir = path / 'Annotations'
    labels_dir = path / 'labels'

    if not labels_dir.exists():
        labels_dir.mkdir()

    if xml_dir.is_dir():
        for xml_file in xml_dir.iterdir():
            tree = ET.parse(str(xml_file))
            root = tree.getroot()
            images_size = root.find('size')

            width = int(images_size.find('width').text)
            height = int(images_size.find('height').text)

            for obj in root.iter('object'):
                cls = obj.find('name').text


if __name__ == '__main__':
    path = r'./testdata'
    voc_parse(path)
