"""
This module defines the pipeline execution for Mixture RAG.
"""

import logging
import os
import warnings

from langfuse import Langfuse
from langfuse.callback import CallbackHandler

from src.configuration.configuration_model import MixtureRAGConfig
from src.constants import evaluation_config
from src.models.mixture_rag import MixtureRAG
from src.utils.evaluation import init_llm_n_metrics, score_output

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")
HOST = os.environ.get("LANGFUSE_HOST")
METRICS = evaluation_config.METRICS


def mixture_rag_pipeline_execution(
    config: MixtureRAGConfig, rag_prompts: list, aggregator_prompt: str, questions: list
):
    """
    Executes the Mixture RAG pipeline.

    Args:
        config (MixtureRAGConfig): The configuration for Mixture RAG.
        prompt (str): The prompt to be used.
        questions (list): A list of questions to be processed.

    Returns:
        None
    """
    logger.info("Starting Mixture RAG pipeline execution")
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

    logger.info("Initializing Mixture RAG")
    mixture_rag = MixtureRAG(config)
    mixture_rag.initialize_base()

    if not mixture_rag.check_vector_store():
        mixture_rag.load_split_ingest_data()
        mixture_rag.save_vector_store()
    else:
        mixture_rag.load_vector_store()

    mixture_rag.create_rag_prompts(rag_prompts)
    mixture_rag.create_aggregator_prompt(aggregator_prompt)
    mixture_rag.initialize_retriever()
    mixture_rag.create_rag_llm_chains()
    chain = mixture_rag.get_aggregator_llm_chain()

    for question in questions:
        logger.info("Processing question: %s", question)
        contexts = [
            context.page_content for context in mixture_rag.retriever.invoke(question)
        ]
        answer = chain.invoke(
            {"question": question, "context": contexts},
            config={"callbacks": [langfuse_handler]},
        ).content
        trace_id = langfuse_handler.get_trace_id()
        score_output(langfuse, trace_id, METRICS, question, contexts, answer)
        print(question, answer, contexts)

    logger.info("Mixture RAG pipeline execution complete")
