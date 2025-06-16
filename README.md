# Local RAG Chatbot
A fully local chatbot that answers questions from your policy documents using retrieval-augmented generation. No internet, no API keys, runs on your GPU (RTX 3050).

## 🚀 Features
- Domain-aware QA using local PDFs
- Fast response with quantized open LLMs
- Lightweight: <4GB VRAM requirement

## Directory Structure
```graphql
policybot/
├── data/                      # PDFs or text docs
├── embeddings/                # Stored FAISS index
├── models/                    # Quantized LLM (GGUF format)
├── app/
│   ├── rag_pipeline.py        # Chunking, embedding, retrieval
│   ├── inference.py           # Local LLM inference using llama.cpp
│   ├── api.py                 # FastAPI endpoints
│   └── ui.py                  # Optional Gradio interface
├── Dockerfile                 # For containerization
├── requirements.txt
└── README.md
```

Here’s the **entire setup section rewritten in clean Markdown** format for your README:

````markdown
### ✅ Setup Steps

#### 1. Install Dependencies
```bash
pip install langchain faiss-cpu sentence-transformers llama-cpp-python fastapi uvicorn
````

---

#### 2. Prepare Documents

* Place your PDFs or text files inside the `data/` directory.
* Use LangChain or LlamaIndex to chunk documents and create a FAISS vector index.

---

#### 3. Set Up Embeddings

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
```

---

#### 4. Download and Load LLM

* Download the quantized `mistral-7B.Q4_0.gguf` model from HuggingFace.
* Place it inside the `models/` directory.
* Load it using `llama-cpp-python`:

```python
from llama_cpp import Llama

llm = Llama(model_path="models/mistral-7B.Q4_0.gguf", n_ctx=2048, n_threads=4)
```

---

#### 5. Build Retrieval Pipeline (`rag_pipeline.py`)

* Embed and store document chunks in a FAISS index.
* On user query, retrieve top-k similar chunks and pass them to the LLM.

---

#### 6. Integrate Local LLM Inference (`inference.py`)

* Use `llama-cpp-python` to run the quantized model locally with low memory.
* Sample interface:

```python
response = llm(f"Context: {retrieved_docs} \n\n Question: {query}")
```

---

#### 7. Expose via FastAPI (`api.py`)

* Create a REST endpoint like:

```
GET /ask?query=your_question
```

* Pipeline: **Query → Retrieve → Generate → Return Answer**

You can now run the API locally and test it using `curl`, Postman, or a simple Gradio/Streamlit UI.


