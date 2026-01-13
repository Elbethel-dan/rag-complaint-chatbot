from tqdm import tqdm
import numpy as np
import pandas as pd
import faiss
import os
import json

EMBEDDING_DIM = 384

class ComplaintVectorStore:
    def __init__(self, index=None, texts=None, metadatas=None):
        self.index = index
        self.texts = texts or []
        self.metadatas = metadatas or []

    @classmethod
    def from_parquet(cls, parquet_path: str, index_path: str, meta_path: str,
                     batch_size: int = 5000, normalize: bool = True):
        """
        Build FAISS index from parquet file in batches and save to disk incrementally.
        """

        if not os.path.exists(parquet_path):
            raise FileNotFoundError(parquet_path)

        df = pd.read_parquet(parquet_path)
        total_rows = len(df)

        # Initialize FAISS index
        index = faiss.IndexFlatIP(EMBEDDING_DIM) if normalize else faiss.IndexFlatL2(EMBEDDING_DIM)

        texts = []
        metadatas = []

        # Ensure directories exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)

        for start in tqdm(range(0, total_rows, batch_size), desc="Building FAISS index"):
            end = min(start + batch_size, total_rows)
            batch = df.iloc[start:end]

            # ---- Embeddings ----
            emb_list = [np.array(emb, dtype="float32") for emb in batch["embedding"].values]
            batch_embeddings = np.vstack(emb_list)
            if normalize:
                faiss.normalize_L2(batch_embeddings)

            index.add(batch_embeddings)

            # ---- Texts ----
            if "chunk_text" in batch.columns:
                texts.extend(batch["chunk_text"].astype(str).tolist())
            else:
                texts.extend([""] * len(batch))

            # ---- Metadata ----
                        # ---- New Metadata Extraction Logic ----
            # We must reach INSIDE the "metadata" column to get the values
            for _, row in batch.iterrows():
                # 1. Grab the dictionary stored in the 'metadata' column
                nested_meta = row.get("metadata", {})

                # 2. Extract our specific fields from that dictionary
                entry = {
                    "complaint_id": nested_meta.get("complaint_id", "N/A"),
                    "product_category": nested_meta.get("product_category", "N/A"),
                    "product": nested_meta.get("product", "N/A"),
                    "issue": nested_meta.get("issue", "N/A"),
                    "sub_issue": nested_meta.get("sub_issue", "N/A"),
                    "company": nested_meta.get("company", "N/A"),
                    "state": nested_meta.get("state", "N/A"),
                    "date_received": nested_meta.get("date_received", "N/A"),
                    "chunk_index": nested_meta.get("chunk_index", 0),
                    "total_chunks": nested_meta.get("total_chunks", 0)
                }
                metadatas.append(entry)


            # ---- Save incrementally (optional, every batch) ----
            if (start // batch_size + 1) % 5 == 0 or end == total_rows:
                # Save FAISS index
                faiss.write_index(index, index_path)

                # Save metadata and texts
                payload = {"texts": texts, "metadatas": metadatas}
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f)

        print(f"✅ FAISS index built and saved to {index_path}")
        print(f"✅ Metadata saved to {meta_path}")

        return cls(index=index, texts=texts, metadatas=metadatas)

    # ---------- Load from disk ----------
    @classmethod
    def load(cls, index_path: str, meta_path: str):
        index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(index=index, texts=payload["texts"], metadatas=payload["metadatas"])

    # ---------- Search ----------
    def search(self, query_embedding: np.ndarray, k: int = 5, normalize: bool = True):
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype("float32")
        if normalize:
            faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "score": float(score),
                "text": self.texts[idx],
                "metadata": self.metadatas[idx]
            })
        return results
