from huggingface_hub import snapshot_download, model_info
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
rev = model_info(MODEL_ID).sha                 
snapshot_download(
    repo_id=MODEL_ID,
    revision=rev,
    local_dir="models/all-MiniLM-L6-v2@"+rev,  
    local_dir_use_symlinks=False
)
print("OK -> models/all-MiniLM-L6-v2@"+rev)