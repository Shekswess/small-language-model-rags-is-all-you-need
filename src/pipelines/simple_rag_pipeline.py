"""
This module defines the pipeline execution for SimpleRAG.
"""

from src.models.simple_rag import SimpleRAG
from src.configuration.configuration_model import SimpleRAGConfig


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
        print(chain.invoke(question))