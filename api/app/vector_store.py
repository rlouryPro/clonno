from typing import List
import numpy as np
import os
from pymilvus import (
connections, FieldSchema, CollectionSchema, DataType, Collection, utility
)
from sentence_transformers import SentenceTransformer


from .config import settings


PK_FIELD = "doc_id"
VEC_FIELD = "embedding"
META_FIELD = "metadata"

class VectorStore:
    def __init__(self):

        # Connect to Milvus
        connections.connect(
        alias="default",
        host=settings.milvus_host,
        port=str(settings.milvus_port),
        )
        self.model = SentenceTransformer(settings.embedding_model)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.collection_name = settings.collection_name
        self.collection = self._ensure_collection()
        self.collection.load()


    def _ensure_collection(self) -> Collection:
        if not utility.has_collection(self.collection_name):
            fields = [
            FieldSchema(name=PK_FIELD, dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name=VEC_FIELD, dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name=META_FIELD, dtype=DataType.JSON),
            ]
            schema = CollectionSchema(fields=fields, description="Documents with vector embeddings")
            coll = Collection(name=self.collection_name, schema=schema)
            # Create HNSW index with Inner Product (cosine via normalized vectors)
            coll.create_index(
            field_name=VEC_FIELD,
            index_params={
            "metric_type": "IP",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64},
            },
            )
            return coll
        else:
            return Collection(self.collection_name)


    def _embed(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return np.array(emb, dtype=np.float32)


    def upsert(self, ids: List[str], texts: List[str], payloads: List[dict | None]):
        vectors = self._embed(texts)
        # Delete existing by PK (safe upsert)
        id_list_literal = ", ".join([f'"{i}"' for i in ids])
        expr = f'{PK_FIELD} in [{id_list_literal}]'
        try:
            self.collection.delete(expr=expr)
        except Exception:
            # ignore if not existing
            pass
        rows = [ids, vectors.tolist(), [p or {} for p in payloads]]
        self.collection.insert(rows)
        self.collection.flush()


    def search(self, query: str, top_k: int = 5, with_payload: bool = True):
        vec = self._embed([query])[0].tolist()
        results = self.collection.search(
        data=[vec],
        anns_field=VEC_FIELD,
        param={"metric_type": "IP", "params": {"ef": 128}},
        limit=top_k,
        output_fields=[PK_FIELD, META_FIELD] if with_payload else [PK_FIELD],
        )
        return results[0] # first (and only) query


    def reset(self):
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            self.collection = self._ensure_collection()
            self.collection.load()


store = VectorStore()