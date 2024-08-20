"""
This module defines the pipeline execution for Simple RAG.
"""

import logging
import os
import warnings

from langfuse import Langfuse
from langfuse.callback import CallbackHandler

from src.configuration.configuration_model import SimpleRAGConfig
from src.constants import evaluation_config
from src.models.simple_rag import SimpleRAG
from src.utils.evaluation import init_llm_n_metrics, score_output

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")
HOST = os.environ.get("LANGFUSE_HOST")
METRICS = evaluation_config.METRICS


def simple_rag_pipeline_execution(
    config: SimpleRAGConfig, prompt: list, questions: list
):
    """
    Executes the Simple RAG pipeline.

    Args:
        config (SimpleRAGConfig): The configuration for Simple RAG.
        prompt (str): The prompt to be used.
        questions (list): A list of questions to be processed.

    Returns:
        None
    """
    logger.info("Starting Simple RAG pipeline execution")
    logger.info("Creating Langfuse client")
    langfuse = Langfuse(
        secret_key=SECRET_KEY,
        public_key=PUBLIC_KEY,
        host=HOST,
    )

    langfuse_handler = CallbackHandler(
        secret_key=SECRET_KEY,
        public_key=PUBLIC_KEY,
        host=HOST,
        session_id=config.experiment_name,
    )

    logger.info("Initializing LLM and metrics for evaluation")
    init_llm_n_metrics(METRICS)

    logger.info("Initializing Simple RAG")
    simple_rag = SimpleRAG(config)
    simple_rag.initialize_base()

    if not simple_rag.check_vector_store():
        simple_rag.load_split_ingest_data()
        simple_rag.save_vector_store()
    else:
        simple_rag.load_vector_store()

    simple_rag.create_prompt(prompt[0])
    simple_rag.initialize_retriever()
    chain = simple_rag.get_retrieval_qa_chain()

    for question in questions:
        logger.info("Processing question: %s", question)
        contexts = [
            context.page_content for context in simple_rag.retriever.invoke(question)
        ]
        answer = chain.invoke(
            question,
            config={"callbacks": [langfuse_handler]},
        )["result"]
        trace_id = langfuse_handler.get_trace_id()
        score_output(langfuse, trace_id, METRICS, question, contexts, answer)
        print(question, answer, contexts)

    logger.info("Simple RAG pipeline execution complete")
