import gradio as gr
import time
from src.rag_pipeline import build_rag_pipeline

# --- 1. Initialize your existing RAG Pipeline ---
# Ensure these paths match your local files
# In app.py
BASE_PATH = "/Users/elbethelzewdie/Downloads/rag-complaint-chatbot/rag-complaint-chatbot"

pipeline = build_rag_pipeline(
    faiss_index_path=f"{BASE_PATH}/faiss.index",  # Changed from faiss_index.bin to faiss.index
    meta_path=f"{BASE_PATH}/metadata.json",
    llm_model_path=f"{BASE_PATH}/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
)


# --- 2. Chat Function (with Streaming and Sources) ---
def predict(message, history):
    # ... (Retrieval happens here) ...
    retrieved_chunks = pipeline.retriever.retrieve(message, k=5)

    # Create the sources summary for display
    sources = "\n\n**Sources:**\n"
    for i, res in enumerate(retrieved_chunks):
        meta = res.get('metadata', {}) # Use .get() for safety
        # Ensure keys exist before accessing
        c_id = meta.get('complaint_id', 'N/A')
        product = meta.get('product', 'N/A')
        issue = meta.get('issue', 'N/A')
        sources += f"- [{c_id}] {product}: {issue}\n"

    # Streaming the response (Requirement: Streaming)
    # Note: We use a generator with 'yield' to stream tokens in Gradio
    full_response = ""

    # We call the generator's internal model to get a stream
    # Mistral v0.3 prompt format
    prompt = f"<s>[INST] Use the context to answer: {message}\n\nCONTEXT:\n{pipeline.generator.build_context(retrieved_chunks)} [/INST]"

    # Using the Llama-CPP model stream directly for the UI
    stream = pipeline.generator.model(
        prompt,
        max_tokens=512,
        temperature=0.0,
        stream=True
    )

    for chunk in stream:
        token = chunk['choices'][0]['text']
        full_response += token
        # Yield the partial response + the static sources list
        yield full_response + sources

# --- Update the UI section of app.py ---
# Minimal configuration to ensure compatibility
demo = gr.ChatInterface(
    fn=predict,
    title="🛡️ CrediTrust Analyst Assistant",
    description="Ask questions based on local customer complaint data.",
    examples=[
        "Why do customers report unexpected fees on credit cards?",
        "What are the common complaints about personal loan interest rates?"
    ]
)

if __name__ == "__main__":
    demo.launch()