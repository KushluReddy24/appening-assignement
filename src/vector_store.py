from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    PINECONE_CLOUD,
    PINECONE_REGION,
    required_env,
)


def get_pinecone_client() -> Pinecone:
    return Pinecone(api_key=required_env("PINECONE_API_KEY"))


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def ensure_index(index_name: str) -> Pinecone:
    client = get_pinecone_client()
    if index_name in client.list_indexes().names():
        dimension = client.describe_index(index_name).dimension
        if dimension != EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"Pinecone index '{index_name}' has dimension {dimension}, but the local "
                f"embedding model needs {EMBEDDING_DIMENSION}. Set "
                "PINECONE_LOCAL_INDEX_NAME to a new index name."
            )
    else:
        client.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    return client


def open_vector_store(index_name: str) -> PineconeVectorStore:
    client = get_pinecone_client()
    if index_name not in client.list_indexes().names():
        raise RuntimeError(
            f"Pinecone index '{index_name}' does not exist. Run `python -m src.ingestion` first."
        )
    dimension = client.describe_index(index_name).dimension
    if dimension != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Pinecone index '{index_name}' has dimension {dimension}; "
            f"the configured local embedding model requires {EMBEDDING_DIMENSION}. "
            "Set PINECONE_LOCAL_INDEX_NAME to the index created by local ingestion."
        )
    return PineconeVectorStore(
        index=client.Index(index_name),
        embedding=get_embeddings(),
    )