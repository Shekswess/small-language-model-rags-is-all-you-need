"""
This module defines the pipeline execution for Mixture RAG.
"""

import os

from langfuse.callback import CallbackHandler

from src.configuration.configuration_model import MixtureRAGConfig
from src.models.mixture_rag import MixtureRAG


def mixture_rag_pipeline_execution(
    config: MixtureRAGConfig, rag_prompts: list, aggregator_prompt: str, questions: list
):
    """
    Executes the SimpleRAG pipeline.

    Args:
        config (SimpleRAGConfig): The configuration for SimpleRAG.
        prompt (str): The prompt to be used.
        questions (list): A list of questions to be processed.

    Returns:
        None
    """

    langfuse_handler = CallbackHandler(
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        host=os.environ["LANGFUSE_HOST"],
        session_id=config.experiment_name,
    )

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
        contexts = [context.page_content for context in mixture_rag.retriever.invoke(question)]
        answer = chain.invoke(
            {"question": question, "context": contexts},
            config={"callbacks": [langfuse_handler]},
        )
        print(question, answer, contexts)