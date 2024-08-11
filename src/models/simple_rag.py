"""Module for the SimpleRAG model."""

import os
import sys
import warnings

from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import BedrockChat, ChatOpenAI
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from .base_simple_rag import SimpleRAGBase

sys.path.append("./src")

from configuration.configuration_model import SimpleRAGConfig

warnings.filterwarnings("ignore")


class SimpleRAG(SimpleRAGBase):
    """SimpleRAG model class."""

    def __init__(self, config: SimpleRAGConfig):
        self.config = config
        self.llm = None
        self.embedder = None
        self.splitter = None
        self.vector_store = None
        self.retriever = None
        self.prompt = None
        self.chain = None
        self.initialize_base()

    def initialize_llm(self):
        """
        Method to initialize the LLM.
        """
        if self.config.llm.provider == "bedrock":
            self.llm = BedrockChat(
                region_name=os.environ["BEDROCK_REGION_NAME"],
                credentials_profile_name=os.environ["BEDROCK_CREDENTIALS_PROFILE_NAME"],
                model_id=self.config.llm.model_spec.model_id,
                model_kwargs=self.config.llm.model_spec.model_kwargs,
            )
        elif self.config.llm.provider == "groq":
            self.llm = ChatGroq(
                model_name=self.config.llm.model_spec.model_name,
                temperature=self.config.llm.model_spec.temperature,
                max_tokens=self.config.llm.model_spec.max_tokens,
                groq_api_key=os.environ["GROQ_API_KEY"],
            )
        elif self.config.llm.provider == "openai":
            self.llm = ChatOpenAI(
                model=self.config.llm.model_spec.model,
                max_tokens=self.config.llm.model_spec.max_tokens,
                temperature=self.config.llm.model_spec.temperature,
                api_key=os.environ["OPENAI_API_KEY"],
            )
        else:
            raise ValueError(f"Invalid LLM provider: {self.config.llm.provider}")

    def initialize_embedder(self):
        """
        Method to initialize the embedding model.
        """
        self.embedder = BedrockEmbeddings(
            region_name=os.environ["BEDROCK_REGION_NAME"],
            credentials_profile_name=os.environ["BEDROCK_CREDENTIALS_PROFILE_NAME"],
            model_id=self.config.embedder.model_id,
            model_kwargs=self.config.embedder.model_kwargs,
        )

    def initialize_splitter(self):
        """
        Method to initialize the text splitter.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunker.chunk_size,
            chunk_overlap=self.config.chunker.chunk_overlap,
        )

    def initialize_base(self):
        """
        Method to initialize the base model.
        """
        self.initialize_llm()
        self.initialize_embedder()
        self.initialize_splitter()

    def load_split_ingest_data(self):
        """
        Method to load, split and ingest data into the vector store.
        """
        global_chunks = []
        for file in os.listdir(self.config.data.path):
            if file.endswith(".pdf"):
                loader = PyMuPDFLoader(os.path.join(self.config.data.path, file))
                docs = loader.load()
                chunks = self.splitter.split_documents(docs)
                global_chunks.extend(chunks)
        self.vector_store = FAISS.from_documents(
            documents=global_chunks, embedding=self.embedder
        )

    def save_vector_store(self):
        """
        Method to save the vector store.
        """
        if not os.path.exists(self.config.vector_store.path):
            os.makedirs(self.config.vector_store.path)
        self.vector_store.save_local(folder_path=self.config.vector_store.path)

    def check_vector_store(self):
        """
        Method to check if the vector store exists.
        """
        chunk_size = self.config.vector_store.path.split("/")[-1].split("_")[-2]
        chunk_overlap = self.config.vector_store.path.split("/")[-1].split("_")[-1]
        if (
            os.path.exists(self.config.vector_store.path)
            and self.config.chunker.chunk_size == int(chunk_size)
            and self.config.chunker.chunk_overlap == int(chunk_overlap)
        ):
            return True
        return False

    def load_vector_store(self):
        """
        Method to load the vector store.
        """
        self.vector_store = FAISS.load_local(
            folder_path=self.config.vector_store.path,
            embeddings=self.embedder,
            allow_dangerous_deserialization=True,
        )

    def initialize_retriever(self):
        """
        Method to initialize the retriever.
        """
        self.retriever = self.vector_store.as_retriever(
            search_type=self.config.retriever.search_type,
            retriever_kwargs=self.config.retriever.retriever_kwargs,
        )

    def create_prompt(self, template: str):
        """
        Method to create the prompt for the LLM.

        Args:
            template (str): The template for the prompt.
        """
        template = template.format(
            system_message=self.config.llm.prompt.system_message,
            user_message=self.config.llm.prompt.user_message,
        )
        self.prompt = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )

    def get_retrieval_qa_chain(self) -> RetrievalQA:
        """
        Method to get the retrieval QA chain.

        Returns:
            RetrievalQA: The retrieval QA chain.
        """
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt, "verbose": True},
        )
        return self.chain
