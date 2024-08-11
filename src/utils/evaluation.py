"""Module for evaluation of chains"""

import logging
import os
import warnings

from langchain_community.chat_models import BedrockChat
from langchain_community.embeddings import BedrockEmbeddings
from langfuse import Langfuse
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics.base import MetricWithEmbeddings, MetricWithLLM
from ragas.run_config import RunConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

LLM_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
LLM_MODEL_KWARGS = {"max_tokens": 4096, "temperature": 0.1}
EMBEDDER_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBEDDER_MODEL_KWARGS = {"dimensions": 512, "normalize": True}


def _wrap_model(
    llm: BedrockChat, embedder: BedrockEmbeddings
) -> tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    """
    Wrap the LLM and Embeddings models for RAGAS metrics.

    Args:
        llm (BedrockChat): LLM model
        embedder (BedrockEmbeddings): Embeddings model

    Returns:
        tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]: Wrapped models
    """
    logger.info("Wrapping models for metrics")
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embedder)


def _init_metric(
    metric, llm: LangchainLLMWrapper, embedder: LangchainEmbeddingsWrapper
) -> None:
    """
    Initialize a single metric with the appropriate models.

    Args:
        metric: Metric
        llm: LangchainLLMWrapper
        embedder: LangchainEmbeddingsWrapper

    Returns:
        None
    """
    logger.info("Initializing metric: %s", metric.name)
    if isinstance(metric, MetricWithLLM):
        logger.info("Setting LLM for metric: %s", metric.name)
        metric.llm = llm
    if isinstance(metric, MetricWithEmbeddings):
        logger.info("Setting Embeddings for metric: %s", metric.name)
        metric.embeddings = embedder
    run_config = RunConfig()
    metric.init(run_config)


def init_llm_n_metrics(
    metrics: list,
    llm_model_id: str = LLM_MODEL_ID,
    llm_model_kwargs: dict = LLM_MODEL_KWARGS,
    embedder_model_id: str = EMBEDDER_MODEL_ID,
    embedder_model_kwargs: dict = EMBEDDER_MODEL_KWARGS,
) -> None:
    """
    Initialize the wrapped models for metrics.

    Args:
        metrics (list): List of metrics
        llm_model_id (str): LLM model ID
        llm_model_kwargs (dict): LLM model keyword arguments
        embedder_model_id (str): Embedder model ID
        embedder_model_kwargs (dict): Embedder model keyword arguments

    Returns:
        None
    """
    logger.info("Initializing LLM and Embedder models")
    llm_eval = BedrockChat(
        region_name=os.environ["BEDROCK_REGION_NAME"],
        credentials_profile_name=os.environ["BEDROCK_CREDENTIALS_PROFILE_NAME"],
        model_id=llm_model_id,
        model_kwargs=llm_model_kwargs,
    )
    embedder_eval = BedrockEmbeddings(
        region_name=os.environ["BEDROCK_REGION_NAME"],
        credentials_profile_name=os.environ["BEDROCK_CREDENTIALS_PROFILE_NAME"],
        model_id=embedder_model_id,
        model_kwargs=embedder_model_kwargs,
    )
    llm, embedder = _wrap_model(llm_eval, embedder_eval)
    logger.info("Initializing metrics")
    for metric in metrics:
        _init_metric(metric, llm, embedder)
    logger.info("Metrics initialized successfully")


def score_output(
    langfuse_client: Langfuse,
    trace_id: str,
    metrics: list,
    question: str,
    context: list[str],
    answer: str,
) -> None:
    """
    Score the output using the metrics.

    Args:
        langfuse_client (Langfuse): Langfuse client
        trace_id (str): Trace ID
        metrics (list): List of metrics
        question (str): Question
        context (list[str]): Context
        answer (str): Answer

    Returns:
        None
    """
    logger.info("Getting scores for the output")
    scores = {
        metric.name: metric.score(
            {"question": question, "contexts": context, "answer": answer}
        )
        for metric in metrics
    }
    logger.info("Scores retrieved successfully")
    logger.info("Writing scores to the trace")
    for score_name, score_value in scores.items():
        langfuse_client.score(trace_id=trace_id, name=score_name, value=score_value)
    logger.info("Scores written to the trace successfully")
