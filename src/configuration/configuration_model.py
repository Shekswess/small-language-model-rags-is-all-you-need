"""Module for defining the structure of the configs for the RAG models."""

from typing import List, Optional, Union, Dict
from pydantic import BaseModel, conint, confloat


class ModelKwargsBedrock(BaseModel):
    """Pydantic model for the model kwargs for the Bedrock model."""

    max_tokens: conint(ge=256, le=8192)
    temperature: confloat(ge=0.0, le=1.0)
    top_k: Optional[conint(ge=0, le=500)] = None
    top_p: Optional[confloat(ge=0.0, le=500.0)] = None
    stop_sequences: Optional[List[str]] = None


class ModelSpecBedrock(BaseModel):
    """Pydantic model for the model spec for the Bedrock model."""

    model_id: Optional[str] = None
    model_kwargs: ModelKwargsBedrock


class ModelSpecGroq(BaseModel):
    """Pydantic model for the model spec for the Groq model"""

    model_name: str
    max_tokens: conint(ge=256, le=8192)
    temperature: confloat(ge=0.0, le=1.0)


class ModelSpecOpenAI(BaseModel):
    """Pydantic model for the model spec for the OpenAI model."""

    model: str
    max_tokens: conint(ge=256, le=8192)
    temperature: confloat(ge=0.0, le=1.0)


class Prompt(BaseModel):
    """Pydantic model for the prompt."""

    system_message: str
    user_message: str


class LLMConfig(BaseModel):
    """Pydantic model for the LLM config."""

    provider: str
    model_spec: Union[ModelSpecBedrock, ModelSpecGroq, ModelSpecOpenAI]
    prompt: Prompt


class EmbedderConfig(BaseModel):
    """Pydantic model for the embedder config."""

    model_id: str
    model_kwargs: Dict[str, Union[int, bool]]


class RetrieverConfig(BaseModel):
    """Pydantic model for the retriever config."""

    search_type: str
    retriever_kwargs: Dict[str, int]


class ChunkerConfig(BaseModel):
    """Pydantic model for the chunker config."""

    chunk_size: int
    chunk_overlap: int


class DataConfig(BaseModel):
    """Pydantic model for the data config for the RAG models."""

    path: str


class VectorStoreConfig(BaseModel):
    """Pydantic model for the vector store config for the RAG models."""

    path: str


class SimpleRAGConfig(BaseModel):
    """Pydantic model for the SimpleRAG config."""

    experiment_name: str
    llm: LLMConfig
    embedder: EmbedderConfig
    retriever: RetrieverConfig
    chunker: ChunkerConfig
    data: DataConfig
    vector_store: VectorStoreConfig


class LayerSpec(BaseModel):
    """Pydantic model for the layer spec."""

    llm: LLMConfig


class Layer(BaseModel):
    """Pydantic model for the layer."""

    layer_type: str
    layer_spec: List[LayerSpec]


class MixtureRAGConfig(BaseModel):
    """Pydantic model for the MixtureRAG config."""

    experiment_name: str
    layers: List[Layer]
    embedder: EmbedderConfig
    retriever: RetrieverConfig
    chunker: ChunkerConfig
    data: DataConfig
    vector_store: VectorStoreConfig
