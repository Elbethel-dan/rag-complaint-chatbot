from llama_cpp import Llama
from typing import List, Dict, Any

class RAGGenerator:
    def __init__(self, model_path: str = "/Users/elbethelzewdie/Downloads/rag-complaint-chatbot/rag-complaint-chatbot/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"):
        # Use the llama-cpp-python library you installed
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=-1, # Ensures Metal GPU use on your MacBook Air
            n_ctx=4096,
            verbose=False
        )

    def build_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        sections = []
        for i, item in enumerate(retrieved_chunks, 1):
            meta = item.get("metadata", {})
            header = f"[Excerpt {i}] Company: {meta.get('company')} | Issue: {meta.get('issue')}"
            sections.append(f"{header}\n{item.get('text','')}")
        return "\n\n".join(sections)

    def generate(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        context = self.build_context(retrieved_chunks)
        # Use your prompt.py template
        prompt = f"<s>[INST] Use the context to answer: {question}\n\nCONTEXT:\n{context} [/INST]"

        output = self.model(prompt, max_tokens=512, temperature=0.0)
        # Note: Fixed the index [0] here which was missing in your text but needed for llama-cpp
        return output['choices'][0]['text'].strip()

# --- ADD THIS PART BELOW ---
def build_generator(llm_model_path: str) -> RAGGenerator:
    """Factory function used by rag_pipeline.py"""
    return RAGGenerator(model_path=llm_model_path)
