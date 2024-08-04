"""Abstract base class for SimpleRAG."""

from abc import ABC, abstractmethod


class SimpleRAGBase(ABC):
    """
    Abstract base class for SimpleRAG.
    """

    def __init__(self):
        pass

    @abstractmethod
    def initialize_llm(self):
        """Method to initialize the LLM."""

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
    def create_prompt(self):
        """Method to create the prompt for the LLM."""

    @abstractmethod
    def get_retrieval_qa_chain(self):
        """Method to get the retrieval QA chain."""
