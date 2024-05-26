from huggingface_hub import snapshot_download

repo_id = "vermouthdky/SimTeG"
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir="Data",
    local_dir_use_symlinks=False,
    allow_patterns=["ogbn-products/all-roberta-large-v1/main/cached_embs/x_embs.pt"],  # "ogbn-arxiv/all-roberta-large-v1/main/cached_embs/x_embs.pt"
)
