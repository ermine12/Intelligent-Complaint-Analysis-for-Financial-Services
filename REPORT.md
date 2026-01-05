# RAG Complaint Chatbot Report

## Task 1: Exploratory Data Analysis
[Analysis of product distribution, narrative length, and missing values to be added after EDA]

## Task 2: Vector Store Implementation
[Sampling strategy, chunking approach, and embedding model choice to be added]

### Sampling Strategy
We used stratified sampling to select 10,000-15,000 records, ensuring proportional representation of the five target product categories.

### Chunking Strategy
We utilized RecursiveCharacterTextSplitter with:
- Chunk Size: 1000 characters
- Chunk Overlap: 200 characters

### Embedding Model
We selected `sentence-transformers/all-MiniLM-L6-v2` because it offers a good balance between performance and speed/size, making it suitable for local execution.
