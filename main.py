from fastapi import FastAPI
from pydantic import BaseModel
from utils.inference import retrieve_docs, generate_answer, load_llm
from utils.rag_pipeline import load_documents
from utils.rag_pipeline import split_documents, build_faiss_index
from logs.logger import CustomLogger
from config import GGUF_MODEL_PATH

logger = CustomLogger.get_logger()
app = FastAPI(title="Local RAG‑LLM API")


class QueryRequest(BaseModel):
    query: str


@app.get("/")
async def root():
    return {"status": "ok"}


@app.post("/ask/")
async def ask(payload: QueryRequest):
    docs = retrieve_docs(payload.query)
    answer = generate_answer(docs, payload.query, llm)
    return {"answer": answer}


@app.post("/update-index/")
async def update_index():
    """Rebuild the FAISS index from current data dir."""
    docs = load_documents()
    chunks = split_documents(docs)
    build_faiss_index(chunks)
    logger.info("FAISS index rebuilt via /update-index/ API call")
    return {
        "status": "Index updated",
        "documents_loaded": len(docs),
        "chunks_created": len(chunks),
    }


@app.on_event("startup")
async def startup_event() -> None:
    """Load the quantised LLM once when the API starts."""
    global llm
    llm = load_llm(GGUF_MODEL_PATH)
    logger.info("LLM loaded at startup")

if __name__ == "__main__":
    pass
