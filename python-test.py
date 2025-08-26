# pip install "pymilvus==2.4.*"
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np

# 1) Connexion au proxy Milvus
connections.connect(alias="default", host="localhost", port="19530")

# 2) (Optionnel) Créer la collection si elle n'existe pas encore
dim = 384                       # dimension de tes embeddings
coll_name = "docs"

if not utility.has_collection(coll_name):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, description="demo")
    coll = Collection(coll_name, schema)

    # Index pour la recherche (tu peux le changer selon tes besoins)
    coll.create_index(
        field_name="embedding",
        index_params={"index_type": "HNSW", "metric_type": "IP", "params": {"M": 16, "efConstruction": 200}},
    )
else:
    coll = Collection(coll_name)

# 3) Données à insérer (exemple)
texts = ["bonjour", "salut", "coucou"]
embeddings = np.random.rand(len(texts), dim).astype("float32").tolist()

# 4) Insertion (ordre = [text, embedding] car 'id' est auto_id)
mr = coll.insert([texts, embeddings])
coll.flush()  # force l’écriture

print("IDs insérés:", mr.primary_keys)