# tests/test_vector_store.py

import numpy as np
import pytest
from unittest.mock import MagicMock
from src.vector_store import ComplaintVectorStore
import io
import json


# -----------------------------
# Fixture: a mock FAISS index
# -----------------------------
class MockIndex:
    def search(self, query_embedding, k):
        # Return dummy scores and indices
        scores = np.array([[0.9, 0.8]])
        indices = np.array([[0, 1]])
        return scores, indices

# -----------------------------
# Fixture: small vector store
# -----------------------------
@pytest.fixture
def mock_vector_store():
    texts = ["Complaint A", "Complaint B"]
    metadatas = [{"id": 1}, {"id": 2}]
    index = MockIndex()
    return ComplaintVectorStore(index=index, texts=texts, metadatas=metadatas)

# -----------------------------
# Test search
# -----------------------------
def test_search_returns_top_k(mock_vector_store):
    query_emb = np.ones(384, dtype="float32")
    results = mock_vector_store.search(query_emb, k=2)

    assert len(results) == 2
    for r in results:
        assert "score" in r
        assert "text" in r
        assert "metadata" in r

    # Scores should match the mock
    assert results[0]["score"] == 0.9
    assert results[1]["score"] == 0.8

# -----------------------------
# Test __init__ stores values
# -----------------------------
def test_init_stores_texts_and_metadata():
    texts = ["A", "B"]
    metas = [{"id": 1}, {"id": 2}]
    cvs = ComplaintVectorStore(index=None, texts=texts, metadatas=metas)

    assert cvs.texts == texts
    assert cvs.metadatas == metas
    assert cvs.index is None

# -----------------------------
# Test load (mock reading JSON)
# -----------------------------
def test_load(monkeypatch):
    # Create a dummy FAISS index
    dummy_index = MockIndex()

    # Create a fake JSON payload for texts and metadatas
    dummy_payload = {"texts": ["X", "Y"], "metadatas": [{"id": 1}, {"id": 2}]}
    dummy_json = json.dumps(dummy_payload)  # Convert dict to JSON string

    # Patch faiss.read_index to return the dummy index
    monkeypatch.setattr("src.vector_store.faiss.read_index", lambda path: dummy_index)

    # Patch open() to return a file-like object (StringIO)
    def mock_open(*args, **kwargs):
        return io.StringIO(dummy_json)

    monkeypatch.setattr("builtins.open", mock_open)

    # Now call the load method — it will use the mocked index and file
    from src.vector_store import ComplaintVectorStore
    store = ComplaintVectorStore.load("dummy.index", "dummy.json")

    # Assertions to verify
    assert store.index == dummy_index
    assert store.texts == ["X", "Y"]
    assert store.metadatas == [{"id": 1}, {"id": 2}]
