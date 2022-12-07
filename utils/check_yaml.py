import yaml
from pathlib import Path


def check_yaml(yaml_path):
    """解析yaml文件"""
    yaml_path = Path(yaml_path)
    suffix = ['.yaml', '.yml']
    if yaml_path.suffix in suffix:
        with open(yaml_path, encoding='utf-8', errors='ignore') as file:
            yaml_dict = yaml.safe_load(file)

        return yaml_dict


if __name__ == '__main__':
    yaml_path = '../models/yolov3.yaml'
    file = check_yaml(yaml_path)
    print(type(file['nc']))


