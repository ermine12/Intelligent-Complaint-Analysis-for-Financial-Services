
import gradio as gr
import os

# Placeholder functions
def retrieve_complaints(query):
    return "This is a placeholder for retrieved complaints based on: " + query

def generate_answer(query):
    return "This is a placeholder for the generated answer."

def rag_interface(query):
    retrieved = retrieve_complaints(query)
    answer = generate_answer(query)
    return answer, retrieved

# Check if data exists
if not os.path.exists('vector_store'):
    warning_msg = "WARNING: Vector store not found. Please run data processing and vector store generation scripts."
else:
    warning_msg = "System Ready."

with gr.Blocks() as demo:
    gr.Markdown("# RAG Complaint Chatbot")
    gr.Markdown(warning_msg)
    
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(label="Enter your query about financial products")
            submit_btn = gr.Button("Submit")
        with gr.Column():
            answer_output = gr.Textbox(label="Answer")
            retrieved_output = gr.Textbox(label="Retrieved Context")
    
    submit_btn.click(fn=rag_interface, inputs=query_input, outputs=[answer_output, retrieved_output])

if __name__ == "__main__":
    demo.launch()
