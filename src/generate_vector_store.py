import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os
import shutil

# Configuration
INPUT_PATH = 'data/filtered_complaints.csv'
VECTOR_STORE_PATH = 'vector_store'
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
SAMPLE_SIZE = 12000  # Target between 10k and 15k
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def generate_vector_store():
    print("Loading filtered dataset...")
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found. Run EDA script first.")
        return

    try:
        df = pd.read_csv(INPUT_PATH, on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    print(f"Loaded {len(df)} records.")

    # Stratified Sampling
    print(f"Sampling {SAMPLE_SIZE} records...")
    # Handle case where dataset is smaller than sample size
    if len(df) < SAMPLE_SIZE:
        print("Dataset smaller than sample size. Using all records.")
        df_sample = df
    else:
        # Stratified sample by Product
        df_sample = df.groupby('Product', group_keys=False).apply(lambda x: x.sample(min(len(x), int(SAMPLE_SIZE * len(x) / len(df)))))
        # If we missed the target slightly due to rounding, sample randomly to adjust or just keep it
        if len(df_sample) > SAMPLE_SIZE:
            df_sample = df_sample.sample(SAMPLE_SIZE)
        print(f"Sampled {len(df_sample)} records.")

    # Chunking
    print("Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    documents = []
    for idx, row in df_sample.iterrows():
        text = row['Consumer complaint narrative']
        metadata = {
            'complaint_id': row.get('Complaint ID', 'N/A'),
            'product': row['Product']
        }
        # Create chunks
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata=metadata))
    
    print(f"Generated {len(documents)} text chunks.")

    # Embedding and Indexing
    print(f"Initializing embedding model {MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    print("Creating FAISS index (this may take a while)...")
    # Batch processing is handled by LangChain usually, but for large datasets monitoring progress is good.
    # We'll just pass all documents.
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save
    print(f"Saving vector store to {VECTOR_STORE_PATH}...")
    vectorstore.save_local(VECTOR_STORE_PATH)
    print("Done.")

if __name__ == "__main__":
    if not os.path.exists('data/filtered_complaints.csv'):
         if os.path.exists('../data/filtered_complaints.csv'):
            os.chdir('..')

    generate_vector_store()
