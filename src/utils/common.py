import os
from box.exceptions import BoxValueError
import yaml
from src.logging import logger
from box import ConfigBox
from pathlib import Path
from typing import Any, List


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its content as a ConfigBox.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)

    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        raise e


def create_directories(path_to_directories: List[Path], verbose: bool = True) -> None:
    """
    Create multiple directories. Logs each directory created.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


def get_size(path: Path) -> str:
    """
    Returns the size of a file in KB.
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~{size_in_kb} KB"



