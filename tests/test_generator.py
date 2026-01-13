import pytest
from unittest.mock import MagicMock, patch

from src.generator import RAGGenerator

# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def dummy_chunks():
    return [
        {"text": "Complaint about product A", "metadata": {"company": "Company A", "issue": "Late delivery"}},
        {"text": "Complaint about product B", "metadata": {"company": "Company B", "issue": "Damaged item"}}
    ]

# -----------------------------
# Test build_context
# -----------------------------
def test_build_context(dummy_chunks):
    gen = RAGGenerator.__new__(RAGGenerator)  # Don't call __init__
    gen.model = MagicMock()
    context = gen.build_context(dummy_chunks)
    assert "[Excerpt 1] Company: Company A | Issue: Late delivery" in context
    assert "[Excerpt 2] Company: Company B | Issue: Damaged item" in context
    assert "Complaint about product A" in context
    assert "Complaint about product B" in context

# -----------------------------
# Test generate
# -----------------------------
def test_generate(dummy_chunks):
    # Patch the Llama model inside RAGGenerator
    with patch("src.generator.Llama") as MockLlama:
        mock_model = MockLlama.return_value
        mock_model.return_value = {"choices": [{"text": "Mocked answer"}]}

        gen = RAGGenerator.__new__(RAGGenerator)  # skip __init__
        gen.model = mock_model

        result = gen.generate("What happened?", dummy_chunks)
        assert result == "Mocked answer"

# -----------------------------
# Test factory
# -----------------------------
def test_build_generator():
    with patch("src.generator.Llama"):
        from src.generator import build_generator
        gen = build_generator("fake/path/model.gguf")
        assert isinstance(gen, RAGGenerator)
