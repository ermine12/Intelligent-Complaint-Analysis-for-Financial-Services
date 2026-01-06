import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# Configuration
VECTOR_STORE_PATH = 'vector_store'
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL = 'microsoft/DialoGPT-medium'  # Using a small conversational model; replace with Mistral/Llama if available

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# Load vector store
if os.path.exists(VECTOR_STORE_PATH):
    vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    raise FileNotFoundError(f"Vector store not found at {VECTOR_STORE_PATH}. Please generate it first.")

# Load LLM pipeline
llm_pipeline = pipeline('text-generation', model=LLM_MODEL, max_length=512, truncation=True)

PROMPT_TEMPLATE = """
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

Context: {context}

Question: {question}

Answer:
"""

def retrieve_complaints(question, k=5):
    """
    Retrieve top-k relevant text chunks for the given question.
    """
    docs = vectorstore.similarity_search(question, k=k)
    return docs

def generate_answer(question, retrieved_docs):
    """
    Generate an answer using the LLM based on the question and retrieved context.
    """
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    # Generate response
    response = llm_pipeline(prompt, max_new_tokens=200, num_return_sequences=1)[0]['generated_text']
    # Extract only the answer part
    answer_start = response.find("Answer:")
    if answer_start != -1:
        answer = response[answer_start + len("Answer:"):].strip()
    else:
        answer = response.strip()
    return answer

def rag_pipeline(question):
    """
    Full RAG pipeline: retrieve and generate.
    """
    retrieved_docs = retrieve_complaints(question)
    answer = generate_answer(question, retrieved_docs)
    sources = [doc.page_content for doc in retrieved_docs]
    return answer, sources