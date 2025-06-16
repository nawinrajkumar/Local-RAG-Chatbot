from huggingface_hub import hf_hub_download
import os


def download_gguf_model(
    repo_id: str = "TheBloke/Mistral-7B-v0.1-GGUF",
    filename: str = "mistral-7b-v0.1.Q4_0.gguf",
    output_dir: str = "../models/"
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    print(f"📦 Downloading {filename} from {repo_id}...")
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=output_dir,
        local_dir=output_dir,
        local_dir_use_symlinks=False
        )
    print(f"✅ Model downloaded to: {model_path}")
    return model_path


if __name__ == "__main__":
    download_gguf_model(
        "TheBloke/Mistral-7B-v0.1-GGUF",
        "mistral-7b-v0.1.Q4_0.gguf",
        "../models"
    )


if __name__ == "__main__":
    pass
