# tests/test_retriever.py

import numpy as np
import pytest
from unittest.mock import MagicMock
from src.retriever import ComplaintRetriever, MiniLMEmbedder

# -----------------------------
# Fixture: mock vector store
# -----------------------------
class MockVectorStore:
    def search(self, query_embedding, k=5, normalize=True):
        # Return dummy search results
        return [
            {"score": 0.95, "text": "Complaint about billing error.", "metadata": {"id": 1}},
            {"score": 0.90, "text": "Dispute rejected due to late submission.", "metadata": {"id": 2}},
        ][:k]

@pytest.fixture
def retriever():
    mock_store = MockVectorStore()
    # Mock embedder that returns a fixed vector
    class MockEmbedder:
        def encode(self, text):
            return np.ones(384, dtype="float32")
    embedder = MockEmbedder()
    return ComplaintRetriever(vector_store=mock_store, embedder=embedder)

# -----------------------------
# Test embedding a question
# -----------------------------
def test_embed_question(retriever):
    q = "Why was my dispute rejected?"
    emb = retriever.embed_question(q)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (384,)  # matches mock embedding

# -----------------------------
# Test retrieval
# -----------------------------
def test_retrieve_top_k(retriever):
    q = "Why was my dispute rejected?"
    results = retriever.retrieve(q, k=2)

    # Check number of results
    assert len(results) == 2

    # Check structure of each result
    for r in results:
        assert "score" in r
        assert "text" in r
        assert "metadata" in r

    # Check that top result is correct
    assert results[0]["score"] >= results[1]["score"]

# -----------------------------
# Test default MiniLM embedder can be instantiated
# -----------------------------
def test_minilm_embedder_init():
    embedder = MiniLMEmbedder()
    sample_text = "Test embedding"
    embedding = embedder.encode(sample_text)
    assert isinstance(embedding, np.ndarray)
    # all-MiniLM-L6-v2 outputs 384-dimensional vectors
    assert embedding.shape[0] == 384
