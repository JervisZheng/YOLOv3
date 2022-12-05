from torch.utils.data import DataLoader, Dataset
import yaml


class YOLODataset(Dataset):
    def __init__(self, root):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


if __name__ == '__main__':
    with open(r'./coco128.yaml', encoding='ascii', errors='ignor') as file:
        yaml = yaml.safe_load(file)
    print(yaml)
