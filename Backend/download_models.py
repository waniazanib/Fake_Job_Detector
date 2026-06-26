from huggingface_hub import snapshot_download

print("Downloading models from HF Hub...")
snapshot_download(
    repo_id="waniazanib/Job_Checking_Model",
    repo_type="model",
    local_dir="/app/models",
    ignore_patterns=["*.md", ".gitattributes"],
)
print("Models downloaded successfully.")