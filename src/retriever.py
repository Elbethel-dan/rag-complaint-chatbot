"""
retriever.py

Reusable retriever module for the complaint vector store.

This module is framework-agnostic and designed to be imported into APIs,
pipelines, notebooks, or RAG systems.

Responsibilities:
- Load an existing ComplaintVectorStore
- Embed user questions with all-MiniLM-L6-v2
- Run top-k similarity search

Public API:
- ComplaintRetriever
- build_retriever(...)
"""

from __future__ import annotations

from typing import List, Dict, Any, Protocol

import numpy as np
from sentence_transformers import SentenceTransformer

from .vector_store import ComplaintVectorStore


# ----------------------------
# Embedder protocol (for extensibility)
# ----------------------------

class Embedder(Protocol):
    def encode(self, text: str) -> np.ndarray: ...


# ----------------------------
# Default embedder
# ----------------------------

class MiniLMEmbedder:
    """Reusable embedder wrapper for all-MiniLM-L6-v2."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> np.ndarray:
        embedding = self.model.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return embedding.astype("float32")


# ----------------------------
# Retriever
# ----------------------------

class ComplaintRetriever:
    """
    Reusable retriever that combines an embedder with a FAISS-backed vector store.

    Parameters
    ----------
    vector_store : ComplaintVectorStore
        Loaded FAISS vector store
    embedder : Embedder
        Any object implementing encode(text) -> np.ndarray
    """

    def __init__(
        self,
        vector_store: ComplaintVectorStore,
        embedder: Embedder,
    ):
        self.vector_store = vector_store
        self.embedder = embedder

    def embed_question(self, question: str) -> np.ndarray:
        """Convert a user question into an embedding vector."""
        return self.embedder.encode(question)

    def retrieve(self, question: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant chunks for a user question.

        Returns
        -------
        List[dict] with keys: score, text, metadata
        """

        query_embedding = self.embed_question(question)
        return self.vector_store.search(query_embedding, k=k, normalize=True)


# ----------------------------
# Factory
# ----------------------------
# Add this to the end of src/retriever.py
def build_retriever(index_path: str, meta_path: str) -> ComplaintRetriever:
    """Factory function used by the RAGPipeline."""
    from .vector_store import ComplaintVectorStore
    v_store = ComplaintVectorStore.load(index_path, meta_path)
    embedder = MiniLMEmbedder() # Ensure this class is defined above
    return ComplaintRetriever(vector_store=v_store, embedder=embedder)
