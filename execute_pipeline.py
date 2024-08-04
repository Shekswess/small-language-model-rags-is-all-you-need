"""
This script executes the appropriate pipeline based on the loaded configuration.
"""

from dotenv import load_dotenv

from src.configuration.configuration_model import MixtureRAGConfig, SimpleRAGConfig
from src.configuration.load_configuration import load_config
from src.constants import prompts, questions
from src.pipelines.simple_rag_pipeline import simple_rag_pipeline_execution

load_dotenv()


CONFIG_PATH = "/home/bojan/Work/mixture-of-rags/config/simple.rag.example.yaml"
PROMPT_CONSTANTS = [prompts.CLAUDE_3_PROMPT_RAG_SIMPLE]
QUESTIONS_CONSTANT = questions.QUESTIONS

if __name__ == "__main__":
    config = load_config(CONFIG_PATH)
    if isinstance(config, MixtureRAGConfig):
        print("MixtureRAGConfig")  # TO DO MixtureRAG pipeline
    elif isinstance(config, SimpleRAGConfig):
        simple_rag_pipeline_execution(config, PROMPT_CONSTANTS, QUESTIONS_CONSTANT)
    else:
        raise ValueError("Invalid configuration type")
