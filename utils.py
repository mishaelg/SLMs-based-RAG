
import json
import yaml


def save_json(data, output_path):
    """
    Save the given data as a JSON file.

    Parameters:
    - data: The data to be saved as JSON.
    - output_path: The path to save the JSON file.

    Returns:
    None
    """
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(input_path):
    """
    Load JSON data from a file.

    Args:
        input_path (str): The path to the JSON file.

    Returns:
        dict: The loaded JSON data.

    """
    with open(input_path, "r") as f:
        data = json.load(f)
    return data


def load_yaml(input_path):
    """
    Load YAML data from the specified input path.

    Args:
        input_path (str): The path to the YAML file.

    Returns:
        dict: The loaded YAML data.
    """
    with open(input_path, "r") as f:
        data = yaml.safe_load(f)
    return data
