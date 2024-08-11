"""Module for the MixtureRAG model."""

import logging
import os
import sys
import warnings

from langchain.document_loaders import PyMuPDFLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import BedrockChat, ChatOpenAI
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_groq import ChatGroq

from .base_mixture_rag import MixtureRAGBase

sys.path.append("./src")

from configuration.configuration_model import MixtureRAGConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


class MixtureRAG(MixtureRAGBase):
    """MixtureRAG model class."""

    def __init__(self, config: MixtureRAGConfig):
        self.config = config
        self.rag_llms = []
        self.aggregator_llm = None
        self.embedder = None
        self.splitter = None
        self.vector_store = None
        self.retriever = None
        self.rag_prompts = []
        self.aggregator_prompt = None
        self.chains = []
        self.chain = None

    def initialize_aggregator_llm(self):
        """
        Method to initialize the aggregator LLM.
        """
        if hasattr(self.config, "layers"):
            for layer in self.config.layers:
                if layer.layer_type == "aggregator":
                    logger.info("Initializing Aggregator LLM")
                    for model in layer.layer_spec:
                        if model.llm.provider == "bedrock":
                            self.aggregator_llm = BedrockChat(
                                region_name=os.environ["BEDROCK_REGION_NAME"],
                                credentials_profile_name=os.environ[
                                    "BEDROCK_CREDENTIALS_PROFILE_NAME"
                                ],
                                model_id=model.llm.model_spec.model_id,
                                model_kwargs=model.llm.model_spec.model_kwargs,
                            )
                            logger.info("Bedrock Aggregator LLM initialized")
                            break
                        elif model.llm.provider == "groq":
                            self.aggregator_llm = ChatGroq(
                                model_name=model.llm.model_spec.model_name,
                                temperature=model.llm.model_spec.temperature,
                                max_tokens=model.llm.model_spec.max_tokens,
                                groq_api_key=os.environ["GROQ_API_KEY"],
                            )
                            logger.info("Groq Aggregator LLM initialized")
                            break
                        elif model.llm.provider == "openai":
                            self.aggregator_llm = ChatOpenAI(
                                model=model.llm.model_spec.model,
                                max_tokens=model.llm.model_spec.max_tokens,
                                temperature=model.llm.model_spec.temperature,
                                api_key=os.environ["OPENAI_API_KEY"],
                            )
                            logger.info("OpenAI Aggregator LLM initialized")
                            break
                        else:
                            logger.error("Invalid provider for aggregator LLM.")
                            raise ValueError("Invalid provider for aggregator LLM.")
        else:
            raise ValueError("No layers found in the configuration.")

    def initialize_rag_llms(self):
        """
        Method to initialize the RAG LLMs.
        """
        if hasattr(self.config, "layers"):
            logger.info("Initializing RAG LLMs")
            for layer in self.config.layers:
                index = self.config.layers.index(layer)
                logger.info(f"Initializing RAG LLM: {index}")
                if layer.layer_type == "rag":
                    for model in layer.layer_spec:
                        if model.llm.provider == "bedrock":
                            rag_llm = BedrockChat(
                                region_name=os.environ["BEDROCK_REGION_NAME"],
                                credentials_profile_name=os.environ[
                                    "BEDROCK_CREDENTIALS_PROFILE_NAME"
                                ],
                                model_id=model.llm.model_spec.model_id,
                                model_kwargs=model.llm.model_spec.model_kwargs,
                            )
                            self.rag_llms.append(rag_llm)
                            logger.info("Bedrock RAG LLM initialized")
                        elif model.llm.provider == "groq":
                            rag_llm = ChatGroq(
                                model_name=model.llm.model_spec.model_name,
                                temperature=model.llm.model_spec.temperature,
                                max_tokens=model.llm.model_spec.max_tokens,
                                groq_api_key=os.environ["GROQ_API_KEY"],
                            )
                            self.rag_llms.append(rag_llm)
                            logger.info("Groq RAG LLM initialized")
                        elif model.llm.provider == "openai":
                            rag_llm = ChatOpenAI(
                                model=model.llm.model_spec.model,
                                max_tokens=model.llm.model_spec.max_tokens,
                                temperature=model.llm.model_spec.temperature,
                                api_key=os.environ["OPENAI_API_KEY"],
                            )
                            self.rag_llms.append(rag_llm)
                            logger.info("OpenAI RAG LLM initialized")
                        else:
                            logger.error("Invalid provider for RAG LLM")
                            raise ValueError("Invalid provider for RAG LLM")
        else:
            logger.error("No layers found in the configuration")
            raise ValueError("No layers found in the configuration.")

    def initialize_embedder(self):
        """
        Method to initialize the embedder model.
        """
        logger.info("Initializing Bedrock Embedder")
        self.embedder = BedrockEmbeddings(
            region_name=os.environ["BEDROCK_REGION_NAME"],
            credentials_profile_name=os.environ["BEDROCK_CREDENTIALS_PROFILE_NAME"],
            model_id=self.config.embedder.model_id,
            model_kwargs=self.config.embedder.model_kwargs,
        )
        logger.info("Bedrock Embedder initialized")


    def initialize_splitter(self):
        """
        Method to initialize the text splitter.
        """
        logger.info("Initializing RecursiveCharacterTextSplitter")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunker.chunk_size,
            chunk_overlap=self.config.chunker.chunk_overlap,
        )
        logger.info("RecursiveCharacterTextSplitter initialized")

    def initialize_base(self):
        """
        Method to initialize the base model.
        """
        logger.info("Initializing Aggregator LLM, RAG LLMs, Embedder and Splitter")
        self.initialize_aggregator_llm()
        self.initialize_rag_llms()
        self.initialize_embedder()
        self.initialize_splitter()
        logger.info("Aggregator LLM, RAG LLMs, Embedder and Splitter initialized")

    def load_split_ingest_data(self):
        """
        Method to load, split and ingest data into the vector store.
        """
        logger.info("Loading, splitting and ingesting data")
        global_chunks = []
        for file in os.listdir(self.config.data.path):
            if file.endswith(".pdf"):
                logger.info(f"Loading, splitting file: {file}")
                loader = PyMuPDFLoader(os.path.join(self.config.data.path, file))
                docs = loader.load()
                chunks = self.splitter.split_documents(docs)
                global_chunks.extend(chunks)
        logger.info("Ingesting data into vector store")
        self.vector_store = FAISS.from_documents(
            documents=global_chunks, embedding=self.embedder
        )
        logger.info("Data ingested into vector store")

    def save_vector_store(self):
        """
        Method to save the vector store.
        """
        logger.info("Saving vector store")
        if not os.path.exists(self.config.vector_store.path):
            logger.info("Creating vector store directory")
            os.makedirs(self.config.vector_store.path)
        self.vector_store.save_local(folder_path=self.config.vector_store.path)
        logger.info("Vector store saved")

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
            logger.info("Vector store exists")
            return True
        logger.info("Vector store does not exist")
        return False

    def load_vector_store(self):
        """
        Method to load the vector store.
        """
        logger.info("Loading vector store")
        self.vector_store = FAISS.load_local(
            folder_path=self.config.vector_store.path,
            embeddings=self.embedder,
            allow_dangerous_deserialization=True,
        )
        logger.info("Vector store loaded")

    def initialize_retriever(self):
        """
        Method to initialize the retriever.
        """
        logger.info("Initializing retriever")
        self.retriever = self.vector_store.as_retriever(
            search_type=self.config.retriever.search_type,
            retriever_kwargs=self.config.retriever.retriever_kwargs,
        )
        logger.info("Retriever initialized")

    def create_rag_prompts(self, templates: list):
        """
        Method to create the prompts for the RAG LLMs.

        Args:
            templates (list): A list of templates for the prompts.
        """
        logger.info("Creating RAG prompts")
        if hasattr(self.config, "layers"):
            for layer in self.config.layers:
                if layer.layer_type == "rag":
                    for model in layer.layer_spec:
                        index = layer.layer_spec.index(model)
                        template = templates[index].format(
                            system_message=model.llm.prompt.system_message,
                            user_message=model.llm.prompt.user_message,
                        )
                        self.rag_prompts.append(
                            PromptTemplate(
                                template=template,
                                input_variables=["context", "question"],
                            )
                        )
            logger.info("RAG prompts created")
        else:
            logger.error("No layers found in the configuration.")
            raise ValueError("No layers found in the configuration.")

    def create_aggregator_prompt(self, template: str):
        """
        Method to create the prompt for the aggregator LLM.

        Args:
            template (str): The template for the prompt.
        """
        logger.info("Creating aggregator prompt")
        if hasattr(self.config, "layers"):
            for layer in self.config.layers:
                if layer.layer_type == "aggregator":
                    for model in layer.layer_spec:
                        template = template.format(
                            system_message=model.llm.prompt.system_message,
                            user_message=model.llm.prompt.user_message,
                        )
                        self.aggregator_prompt = PromptTemplate(
                            template=template,
                            input_variables=["output_1", "output_2", "output_3"],
                        )
            logger.info("Aggregator prompt created")
        else:
            logger.error("No layers found in the configuration")
            raise ValueError("No layers found in the configuration.")

    def create_rag_llm_chains(self):
        """
        Method to get the RAG LLM chain.
        """
        logger.info("Creating RAG LLM chains")
        for rag_llm in self.rag_llms:
            chain = (
                RunnablePassthrough(
                    context=RunnablePassthrough(), question=RunnablePassthrough()
                )
                | self.rag_prompts[self.rag_llms.index(rag_llm)]
                | rag_llm
            )
            self.chains.append(chain)
        logger.info("RAG LLM chains created")

    def get_aggregator_llm_chain(self) -> RunnableParallel:
        """
        Method to get the aggregator LLM chain.

        Returns:
            RunnableParallel: The aggregator LLM chain.
        """
        logger.info("Creating aggregator LLM chain")
        self.chain = (
            RunnableParallel(
                output_1=self.chains[0],
                output_2=self.chains[1],
                output_3=self.chains[2],
            )
            | self.aggregator_prompt
            | self.aggregator_llm
        )
        logger.info("Aggregator LLM chain created")
        return self.chain
