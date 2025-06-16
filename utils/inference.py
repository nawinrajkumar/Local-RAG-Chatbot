from typing import List
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from llama_cpp import Llama

from logs.logger import CustomLogger
logger = CustomLogger.get_logger()


def load_vectorstore(index_path: str = "embeddings/index") -> FAISS:
    """Load FAISS index from disk."""
    logger.info(f"Loading FAISS index from '{index_path}'")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(index_path, 
                            embeddings, 
                            allow_dangerous_deserialization=True)


def retrieve_docs(query: str, k: int = 3) -> List[Document]:
    """Retrieve top-k similar documents for a given query."""
    logger.info(f"Retrieving top {k} documents for query: '{query}'")
    db = load_vectorstore()
    return db.similarity_search(query, k=k)


def load_llm(model_path: str = "models/mistral-7B.Q4_0.gguf") -> Llama:
    """Load quantized LLM using llama-cpp-python."""
    logger.info(f"Loading LLM model from '{model_path}'")
    return Llama(model_path=model_path, n_ctx=2048, n_threads=4)


def generate_answer(
    context_docs: List[Document], query: str, llm: Llama
) -> str:
    """Generate answer from context documents and user query."""
    logger.info("Generating response using context and LLM")
    context = "\n".join([doc.page_content for doc in context_docs])
    prompt = (
        f"Answer the question based on the following context:\n\n{context}"
        f"\n\nQuestion: {query}\nAnswer:"
    )
    output = llm(prompt)
    return output["choices"][0]["text"].strip()


if __name__ == "__main__":
    pass
