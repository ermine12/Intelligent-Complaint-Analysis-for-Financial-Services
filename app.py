
import gradio as gr
import os
from src.rag_pipeline import rag_pipeline

def rag_interface(query):
    try:
        answer, sources = rag_pipeline(query)
        sources_text = "\n\n**Sources:**\n" + "\n".join([f"- {src[:200]}..." for src in sources])
        return answer + sources_text
    except Exception as e:
        return f"Error: {str(e)}", ""

# Check if data exists
if not os.path.exists('vector_store'):
    warning_msg = "WARNING: Vector store not found. Please run data processing and vector store generation scripts."
else:
    warning_msg = "System Ready."

with gr.Blocks() as demo:
    gr.Markdown("# RAG Complaint Chatbot")
    gr.Markdown(warning_msg)

    query_input = gr.Textbox(label="Enter your query about financial products")
    submit_btn = gr.Button("Submit")
    answer_output = gr.Textbox(label="Answer and Sources", lines=10)
    clear_btn = gr.Button("Clear")

    submit_btn.click(fn=rag_interface, inputs=query_input, outputs=answer_output)
    clear_btn.click(fn=lambda: ("", ""), inputs=[], outputs=[query_input, answer_output])

if __name__ == "__main__":
    demo.launch()
