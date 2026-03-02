import os
import json
import faiss
import numpy as np

class FaissStore:
    def __init__(self, index_path: str, meta_path: str, dim: int):
        self.index_path = index_path
        self.meta_path = meta_path
        self.dim = dim

        # FAISS uses volatile storage(RAM); persist() writes index and metadata to disk
        self.index = faiss.IndexFlatIP(dim) # cosine similarity: inner product after normalization
        self.meta = []  # list[dict] each corresponds to a vector

        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

    def add(self, vectors: np.ndarray, metadatas: list[dict]):
        assert vectors.ndim == 2 and vectors.shape[1] == self.dim
        self.index.add(vectors)
        self.meta.extend(metadatas)

    def search(self, query_vec: np.ndarray, top_k: int = 4):
        assert query_vec.shape == (1, self.dim)
        scores, ids = self.index.search(query_vec, top_k) # retrieve: faiss.IndexFlatIP.search()
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            results.append({"score": float(score), "metadata": self.meta[idx]})
        return results

    def persist(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)
