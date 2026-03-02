from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        # Returns (N, D) normalized float32 embeddings for cosine similarity search
        emb = self.model.encode(texts, normalize_embeddings=True)
        return emb.astype("float32")
