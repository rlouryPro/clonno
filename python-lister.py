from pymilvus import connections, Collection, utility

connections.connect(host="localhost", port="19530")  # ou "milvus" selon ton réseau
coll = Collection("docs")        # ta collection
print("Collections :", utility.list_collections())
print("Nb d'entités :", coll.num_entities)

coll.load()  # charge en mémoire pour requêtes
rows = coll.query(
    expr="id >= 0",              # récupère “les premières” lignes
    output_fields=["id", "text"],# évite d'imprimer les gros vecteurs
    limit=5
)
for r in rows:
    print(r)     