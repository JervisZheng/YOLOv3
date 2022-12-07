import yaml


def check_yaml(yaml_path):
    with open(yaml_path, encoding='ascii', errors='ignore') as file:
        yaml = yaml.