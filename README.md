# Vector Search API — Milvus Edition


Milvus (standalone) + FastAPI pour indexer/rechercher des textes via embeddings.


## Démarrage


```bash
cp .env.example .env
# Lancer Milvus + API
docker compose up --build


# API: http://localhost:8000/docs
# Milvus gRPC: localhost:19530 (pymilvus)
# MinIO console (optionnel): http://localhost:9001