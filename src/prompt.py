"""
prompt.py

Prompt templates for the RAG system.
Centralizes all prompt formatting for consistency and easy modification.
"""

# ----------------------------
# RAG Prompt Template
# ----------------------------

RAG_ANALYST_PROMPT: str = """You are a financial analyst assistant for CrediTrust.

Your task is to answer questions about customer complaints using ONLY the information provided in the retrieved complaint excerpts below.
Base your answer strictly on the context. Do NOT use outside knowledge, assumptions, or speculation.

If the context does not contain enough information to answer the question, respond with:
"I do not have enough information in the provided complaints to answer this question."

When the information is available, provide a clear, concise, and factual answer written in a professional analyst tone.

Context:
{context}

Question:
{question}

Answer:
"""


# ----------------------------
# Helper
# ----------------------------


def format_rag_prompt(context: str, question: str) -> str:
    """
    Format the RAG analyst prompt with retrieved context and user question.
    """
    return RAG_ANALYST_PROMPT.format(context=context, question=question)
