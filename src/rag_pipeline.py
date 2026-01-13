"""
rag_pipeline.py

Reusable RAG orchestrator module for notebooks and scripts.

Features:
- Loads FAISS vector store and metadata
- Initializes retriever
- Initializes CPU-friendly Mistral generator
- Provides a single run(question, k) method

Designed to be imported and used in Jupyter notebooks, pipelines, or scripts.
"""

from typing import List, Dict, Any

from .retriever import build_retriever, ComplaintRetriever
from .generator import build_generator, RAGGenerator


class RAGPipeline:
    """
    End-to-end reusable RAG pipeline.

    Parameters
    ----------
    faiss_index_path : str
        Path to FAISS index
    meta_path : str
        Path to metadata JSON
    llm_model_path : str
        Path to quantized GGUF Mistral model
    """

    def __init__(
        self,
        faiss_index_path: str,
        meta_path: str,
        llm_model_path: str,
    ):
        self.retriever: ComplaintRetriever = build_retriever(faiss_index_path, meta_path)
        self.generator: RAGGenerator = build_generator(llm_model_path)

    def run(self, question: str, k: int = 5) -> str:
        """
        Retrieve top-k relevant chunks and generate a grounded answer.

        Parameters
        ----------
        question : str
            User question
        k : int
            Number of chunks to retrieve

        Returns
        -------
        str
            LLM-generated answer
        """
        retrieved_chunks: List[Dict[str, Any]] = self.retriever.retrieve(question, k=k)
        return self.generator.generate(question, retrieved_chunks)


# ----------------------------
# Convenience factory
# ----------------------------

def build_rag_pipeline(
    faiss_index_path: str,
    meta_path: str,
    llm_model_path: str,
) -> RAGPipeline:
    """Build and return a reusable RAGPipeline instance."""
    return RAGPipeline(faiss_index_path, meta_path, llm_model_path)


