from typing import Union

import yaml
from pydantic import ValidationError

from .configuration_model import MixtureRAGConfig, SimpleRAGConfig


def load_config(file_path: str) -> Union[MixtureRAGConfig, SimpleRAGConfig]:
    """
    Loading the configuration from a YAML file for the RAG models.

    Args:
        file_path(str): Path to the YAML file containing the configuration.

    Returns:
        Union[MixtureRAGConfig, SimpleRAGConfig]: The configuration object.

    Raises:
        ValueError: If the configuration is invalid.
    """
    rag_type = file_path.split("/")[-1].split(".")[0]
    with open(file_path, "r") as file:
        try:
            config = yaml.safe_load(file)
            if rag_type == "mixture":
                return MixtureRAGConfig(**config)
            elif rag_type == "simple":
                return SimpleRAGConfig(**config)
            else:
                raise ValueError(f"Invalid RAG type: {rag_type}")
        except ValidationError as error:
            raise ValueError(f"Invalid configuration: {error}")
        except Exception as error:
            raise ValueError(f"Error loading configuration: {error}")
