from pydantic_settings import BaseSettings
from pydantic import Field
import os


class Settings(BaseSettings):
    milvus_host: str = Field(default="localhost", alias="MILVUS_HOST")
    milvus_port: int = Field(default=19530, alias="MILVUS_PORT")
    collection_name: str = Field(default="documents", alias="COLLECTION_NAME")
    embedding_model: str = os.getenv("EMBEDDING_MODEL_PATH", "models/all-MiniLM-L6-v2")

class Config:
    env_file = ".env"
    extra = "ignore"


settings = Settings()