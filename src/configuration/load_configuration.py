"""Module to load the configuration for the RAG models."""

import logging
import warnings
from typing import Union

import yaml
from pydantic import ValidationError

from .configuration_model import MixtureRAGConfig, SimpleRAGConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


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
    logger.info("Loading configuration from: %s", file_path)
    rag_type = file_path.split("/")[-1].split(".")[0]
    logger.info("RAG type: %s", rag_type)
    with open(file_path, "r") as file:
        try:
            config = yaml.safe_load(file)
            if rag_type == "mixture":
                logger.info("Loading MixtureRAG configuration")
                return MixtureRAGConfig(**config)
            elif rag_type == "simple":
                logger.info("Loading SimpleRAG configuration")
                return SimpleRAGConfig(**config)
            else:
                logger.error("Invalid RAG type: %s", rag_type)
                raise ValueError(f"Invalid RAG type: {rag_type}")
        except ValidationError as error:
            logger.error("Invalid configuration: %s", error)
            raise ValueError(f"Invalid configuration: {error}")
        except Exception as error:
            logger.error("Error loading configuration: %s", error)
            raise ValueError(f"Error loading configuration: {error}")
