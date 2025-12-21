from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Changed this line
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Load PDF documents from the "data" folder
documents = []
for file in os.listdir("data"):
    # Only process PDF files
    if file.endswith(".pdf"):
        # Reads the PDF Files
        loader = PyPDFLoader(f"data/{file}")
        documents.extend(loader.load())

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    # Each chunk shares 200 characters with previous to maintain context
    chunk_size=1000,
    chunk_overlap=200
)
# Create a list of smaller document chunks
chunks = splitter.split_documents(documents)

# Create vector from chunks
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Store the vectors in a database to search quickly
vectorstore = FAISS.from_documents(chunks, embeddings)

vectorstore.save_local("faiss_index")
print("Vector database created")