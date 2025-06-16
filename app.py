import gradio as gr
import requests

API_URL = "http://localhost:8000/ask/"


def rag_chatbot(query: str) -> str:
    try:
        response = requests.post(API_URL, json={"query": query})
        if response.status_code == 200:
            return response.json().get("answer", "No answer returned.")
        return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Failed to connect to API: {str(e)}"


gr.Interface(
    fn=rag_chatbot,
    inputs=gr.Textbox(label="Enter your question"),
    outputs=gr.Textbox(label="Answer"),
    title="RAG Assistant",
    description="Ask questions based on your custom documents."
).launch()
