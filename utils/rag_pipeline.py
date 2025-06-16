# rag_pipeline.py

import os
from typing import List
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from logs.logger import CustomLogger
from config import DATA_DIR

logger = CustomLogger.get_logger()


def load_documents(data_dir: str = "data/") -> List[Document]:
    """Load PDF and TXT files from a directory into Document objects."""
    docs: List[Document] = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath)
        else:
            continue
        docs.extend(loader.load())
    logger.info(f"Loaded {len(docs)} documents from '{data_dir}'")
    return docs


def split_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Document]:
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split documents into {len(chunks)} chunks")
    return chunks


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return sentence-transformer embedding model."""
    logger.info("Loading HuggingFace embedding model")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def build_faiss_index(
    docs: List[Document],
    index_path: str = "embeddings/index"
) -> None:
    """Embed documents and save them as a FAISS index."""
    logger.info("Generating embeddings and building FAISS index")
    embeddings = get_embedding_model()
    vector_db = FAISS.from_documents(docs, embeddings)
    vector_db.save_local(index_path)
    logger.info(f"FAISS index saved to '{index_path}'")


if __name__ == "__main__":
    logger.info("Starting RAG pipeline")

    raw_docs = load_documents(DATA_DIR)
    chunks = split_documents(raw_docs)
    build_faiss_index(chunks)

    logger.info("RAG pipeline completed successfully")
