import numpy as np
from pymilvus import connections, Collection, utility

# adapte host/port si besoin
connections.connect(
    alias="default",
    host="127.0.0.1",
    port="19530",
    timeout=60,
    # si l’auth est activée chez toi :
    # user="root", password="Milvus"
)

coll = Collection("docs")
coll.load()

# vecteur de requête (exemple aléatoire ; remplace par ton embedding réel)
dim = 384
qvec = np.random.rand(1, dim).astype("float32").tolist()

hits = coll.search(
    data=qvec,
    anns_field="embedding",
    param={"metric_type": "IP", "params": {"ef": 64}},  # adapté à HNSW
    limit=3,
    output_fields=["id", "text"]
)
for hit in hits:
    for r in hit:
        print(r.id, r.distance, r.entity.get("text"))