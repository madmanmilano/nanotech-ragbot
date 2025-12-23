import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="UCB NanoTech Research Assistant",
    page_icon="🔬"
)

# Berkeley logo at top
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/University_of_California%2C_Berkeley_logo.svg/1200px-University_of_California%2C_Berkeley_logo.svg.png", width=120)

# Set up page
st.title("🔬 UCB NanoTech Chatbot")
st.write("Ask questions about the research documents")
st.divider()

# Load resources (cached so it only loads once)
@st.cache_resource
def load_chatbot():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Build vector database from PDFs
    documents = []
    # Load PDF documents from the "data" folder
    for file in os.listdir("data"):
        # Only process PDF files
        if file.endswith(".pdf"):
            # Reads the PDF Files
            loader = PyPDFLoader(f"data/{file}")
            documents.extend(loader.load())
    
    # Split documents into chunks
    # Each chunk shares 200 characters with previous to maintain context
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # Create a list of smaller document chunks
    chunks = splitter.split_documents(documents)
    # Store the vectors in a database to search quickly
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Use Groq LLM for Q&A
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7
    )
    
    # Create Q&A chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )
    return qa_chain

qa_chain = load_chatbot()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if prompt := st.chat_input("Ask a question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"query": prompt})
            response = result['result']
            st.write(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})