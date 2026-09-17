from __future__ import annotations

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import settings
from .document_loader import load_and_split_documents


class LocalDefaultEmbeddings(Embeddings):
    """Local, offline embedding generator using Chroma's default ONNX model (no external API key needed)."""

    def __init__(self) -> None:
        import chromadb.utils.embedding_functions as ef

        self._ef = ef.DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, doc)) for doc in self._ef(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [float(x) for x in self._ef([text])[0]]


def get_embeddings() -> Embeddings:
    """Return local offline embeddings without requiring any OpenAI key."""
    return LocalDefaultEmbeddings()


def get_vector_store() -> Chroma:
    store = Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
    )
    try:
        if store._collection.count() == 0:
            store, _ = rebuild_vector_store()
    except Exception:
        pass
    return store


def rebuild_vector_store() -> tuple[Chroma, int]:
    import chromadb

    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    try:
        client.delete_collection(settings.collection_name)
    except Exception:
        pass

    chunks = load_and_split_documents()
    if not chunks:
        raise RuntimeError("No supported documents were found in data/policies.")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=settings.collection_name,
        client=client,
    )
    return vector_store, len(chunks)


def retrieve(query: str, k: int | None = None) -> list[Document]:
    vector_store = get_vector_store()
    return vector_store.similarity_search(query, k=k or settings.top_k)
