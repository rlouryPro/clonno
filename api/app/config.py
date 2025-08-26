from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    milvus_host: str = Field(default="localhost", alias="MILVUS_HOST")
    milvus_port: int = Field(default=19530, alias="MILVUS_PORT")
    collection_name: str = Field(default="documents", alias="COLLECTION_NAME")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")


class Config:
    env_file = ".env"
    extra = "ignore"


settings = Settings()