"""
This script executes the appropriate pipeline based on the loaded configuration.
"""
import logging
import warnings

from dotenv import load_dotenv

from src.configuration.configuration_model import (MixtureRAGConfig,
                                                   SimpleRAGConfig)
from src.configuration.load_configuration import load_config
from src.constants import prompts, questions
from src.pipelines.mixture_rag_pipeline import mixture_rag_pipeline_execution
from src.pipelines.simple_rag_pipeline import simple_rag_pipeline_execution

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

CONFIG_PATH = "/home/bojan/Work/mixture-of-rags/config/mixture.rag.example.yaml"
PROMPT_CONSTANTS = [
    prompts.CLAUDE_3_PROMPT_RAG_SIMPLE,
    prompts.CLAUDE_3_PROMPT_RAG_SIMPLE,
    prompts.CLAUDE_3_PROMPT_RAG_SIMPLE,
]
PROMPT_AGGREGATOR_CONSTANT = prompts.CLAUDE_3_MIXTURE_RAG
QUESTIONS_CONSTANT = questions.QUESTIONS

if __name__ == "__main__":
    logger.info("Loading configuration")
    config = load_config(CONFIG_PATH)
    if isinstance(config, MixtureRAGConfig):
        logger.info("Executing MixtureRAG pipeline")
        mixture_rag_pipeline_execution(
            config, PROMPT_CONSTANTS, PROMPT_AGGREGATOR_CONSTANT, QUESTIONS_CONSTANT
        )
    elif isinstance(config, SimpleRAGConfig):
        logger.info("Executing SimpleRAG pipeline")
        simple_rag_pipeline_execution(config, PROMPT_CONSTANTS, QUESTIONS_CONSTANT)
    else:
        logger.error("Invalid configuration type")
        raise ValueError("Invalid configuration type")
