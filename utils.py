
import json
import yaml


def save_json(data, output_path):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(input_path):
    with open(input_path, "r") as f:
        data = json.load(f)
    return data


def load_yaml(input_path):
    with open(input_path, "r") as f:
        data = yaml.safe_load(f)
    return data
