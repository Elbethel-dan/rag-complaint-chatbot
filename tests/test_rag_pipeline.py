import pytest
from unittest.mock import MagicMock, patch

from src.rag_pipeline import RAGPipeline, build_rag_pipeline

# -----------------------------
# Dummy data for retrieval
# -----------------------------
dummy_chunks = [
    {"text": "Complaint about product A", "metadata": {"company": "Company A", "issue": "Late delivery"}},
    {"text": "Complaint about product B", "metadata": {"company": "Company B", "issue": "Damaged item"}}
]

# -----------------------------
# Test RAGPipeline.run with mocks
# -----------------------------
def test_rag_pipeline_run():
    # Mock the retriever and generator
    with patch("src.rag_pipeline.build_retriever") as mock_build_retriever, \
         patch("src.rag_pipeline.build_generator") as mock_build_generator:

        # Create dummy retriever
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = dummy_chunks
        mock_build_retriever.return_value = mock_retriever

        # Create dummy generator
        mock_generator = MagicMock()
        mock_generator.generate.return_value = "Mocked answer"
        mock_build_generator.return_value = mock_generator

        # Initialize pipeline
        pipeline = RAGPipeline("dummy.index", "dummy.json", "dummy_model.gguf")

        # Run the pipeline
        answer = pipeline.run("What happened?", k=2)

        # Assertions
        mock_retriever.retrieve.assert_called_once_with("What happened?", k=2)
        mock_generator.generate.assert_called_once_with("What happened?", dummy_chunks)
        assert answer == "Mocked answer"

# -----------------------------
# Test factory function
# -----------------------------
def test_build_rag_pipeline():
    with patch("src.rag_pipeline.build_retriever"), patch("src.rag_pipeline.build_generator"):
        pipeline = build_rag_pipeline("dummy.index", "dummy.json", "dummy_model.gguf")
        from src.rag_pipeline import RAGPipeline
        assert isinstance(pipeline, RAGPipeline)
