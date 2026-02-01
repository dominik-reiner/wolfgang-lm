import os
import sys
from huggingface_hub import hf_hub_download


def download_model():
    repo_id = os.getenv("HF_MODEL_ID")
    filename = os.getenv("HF_FILENAME")
    local_dir = os.getenv("MODEL_DIR")

    if not repo_id or not filename or not local_dir:
        print("Error: HF_MODEL_ID, HF_FILENAME, and MODEL_DIR must be set.")
        sys.exit(1)

    print(f"Downloading {filename} from {repo_id} to {local_dir}...")
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_model()
