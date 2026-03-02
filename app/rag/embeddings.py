from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        # (N, D) float32
        emb = self.model.encode(texts, normalize_embeddings=True)
        return emb.astype("float32")

# SentenceTransformer prepares engine for the embedding model
# Normalizing for similarity comparison. (make vectors' length 1 by dividing vector by magnitude)