from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List


from .schemas import UpsertRequest, SearchRequest, SearchResponse, SearchHit
from .vector_store import store


app = FastAPI(title="Vector Search API (Milvus)", version="1.0.0")


app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upsert")
async def upsert(req: UpsertRequest):
    if not req.items:
        raise HTTPException(status_code=400, detail="items cannot be empty")
    ids = [it.id for it in req.items]
    texts = [it.text for it in req.items]
    payloads = [it.metadata for it in req.items]
    store.upsert(ids, texts, payloads)
    return {"upserted": len(ids)}


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    res = store.search(query=req.query, top_k=req.top_k, with_payload=req.with_payload)
    hits: List[SearchHit] = [
        SearchHit(
            id=str(hit.entity.get("doc_id")),
            score=float(hit.distance),  # IP similarity on normalized vectors
            metadata=(hit.entity.get("metadata") if req.with_payload else None),
        )
        for hit in res
    ]
    return SearchResponse(hits=hits)


@app.delete("/reset")
async def reset():
    store.reset()
    return {"reset": True}