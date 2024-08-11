"""
This module defines the pipeline execution for SimpleRAG.
"""

import os

from langfuse.callback import CallbackHandler

from src.configuration.configuration_model import SimpleRAGConfig
from src.models.simple_rag import SimpleRAG


def simple_rag_pipeline_execution(config: SimpleRAGConfig, prompt: list, questions: list):
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

    questions = questions[1:2]
    for question in questions:
        contexts = [context.page_content for context in simple_rag.retriever.invoke(question)]
        answer = chain.invoke(
            question,
            config={"callbacks": [langfuse_handler]},
        )['result']
        print(question, answer, contexts)