from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "rag-backend"
    data_dir: str = "data"
    index_dir: str = "data/index"
    raw_dir: str = "data/raw"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 4

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

# embedding_model selection depends on:
# 1) Semantic difficulty
# 2) dataset size
# 3) latency requirements