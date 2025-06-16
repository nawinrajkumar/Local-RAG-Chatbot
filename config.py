import os

# Root directory is two levels up from this file (scripts/app/utils)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# Core folders
DATA_DIR = os.path.join(ROOT_DIR, "data")
EMBEDDINGS_DIR = os.path.join(ROOT_DIR, "embeddings", "index")
MODELS_DIR = os.path.join(ROOT_DIR, "models")  # store .gguf here
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Model configuration
GGUF_MODEL_NAME = "mistral-7b-v0.1.Q4_0.gguf"
GGUF_REPO_ID = "TheBloke/Mistral-7B-v0.1-GGUF"
GGUF_MODEL_PATH = os.path.join(MODELS_DIR, GGUF_MODEL_NAME)

# Embedding / retrieval configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

if __name__ == "__main__":
    pass
