import gradio as gr
from utils.inference import retrieve_docs, load_llm, generate_answer
from logs.logger import CustomLogger

logger = CustomLogger.get_logger()
llm = load_llm()


def rag_chatbot(query: str) -> str:
    logger.info(f"UI received query: {query}")
    docs = retrieve_docs(query)
    answer = generate_answer(docs, query, llm)
    logger.info("Returning answer from UI")
    return answer


gr.Interface(
    fn=rag_chatbot,
    inputs=gr.Textbox(label="Enter your question"),
    outputs=gr.Textbox(label="Answer"),
    title="RAG Assistant",
    description="Ask questions based on your custom documents."
).launch()
