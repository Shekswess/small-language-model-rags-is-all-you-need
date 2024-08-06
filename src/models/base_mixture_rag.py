"""Abstract base class for MixtureRAG."""

from abc import ABC, abstractmethod


class MixtureRAGBase(ABC):
    """
    Abstract base class for MixtureRAG.
    """

    def __init__(self):
        pass

    @abstractmethod
    def initialize_agregator_llm(self):
        """Method to initialize the aggregator LLM."""

    @abstractmethod
    def initialize_rag_llms(self):
        """Method to initialize the RAG LLMs."""

    @abstractmethod
    def initialize_embedder(self):
        """Method to initialize the embedder model."""

    @abstractmethod
    def initialize_splitter(self):
        """Method to initialize the text splitter."""

    @abstractmethod
    def initialize_base(self):
        """Method to initialize the base model."""

    @abstractmethod
    def load_split_ingest_data(self):
        """Method to load, split and ingest data into the vector store."""

    @abstractmethod
    def save_vector_store(self):
        """Method to save the vector store."""

    @abstractmethod
    def check_vector_store(self):
        """Method to check if the vector store exists."""

    @abstractmethod
    def load_vector_store(self):
        """Method to load the vector store."""

    @abstractmethod
    def initialize_retriever(self):
        """Method to initialize the retriever."""

    @abstractmethod
    def create_rag_prompts(self):
        """Method to create the prompts for the RAG LLMs."""

    @abstractmethod
    def create_aggregator_prompt(self):
        """Method to create the prompt for the aggregator LLM."""

    @abstractmethod
    def get_retrieval_qa_chains(self):
        """Method to get the retrieval QA chains for the RAG LLMs."""

    @abstractmethod
    def get_aggregator_llm_chain(self):
        """Method to get the aggregator LLM chain."""
