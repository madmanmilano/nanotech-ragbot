import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
import pickle

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found! Please check your .env file.")
    st.stop()

# ------------------------------
# Streamlit page config
# ------------------------------
st.set_page_config(
    page_title="UCB NanoTech Research Assistant",
    page_icon="🔬",
    layout="wide"
)

# Berkeley logo at top
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/University_of_California%2C_Berkeley_logo.svg/1200px-University_of_California%2C_Berkeley_logo.svg.png",
        width=120
    )

# Page title
st.title("🔬 UCB NanoTech Chatbot")
st.write("Ask questions about the research documents")

# ------------------------------
# User settings
# ------------------------------
with st.expander("⚙️ Settings"):
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider(
            "Response Creativity", 0.0, 1.0, 0.7, 0.1,
            help="Lower = more focused, Higher = more creative"
        )
    with col2:
        num_sources = st.slider(
            "Number of Sources", 1, 5, 3,
            help="How many document chunks to use"
        )

st.divider()

# ------------------------------
# Load chatbot with caching
# ------------------------------
@st.cache_resource
def load_chatbot(temp, k, key):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load existing FAISS index if it exists
    vectorstore = None
    processed_files = set()
    
    if os.path.exists("faiss_index"):
        vectorstore = FAISS.load_local("faiss_index", embeddings)
        # Load processed files list if exists
        processed_path = os.path.join("faiss_index", "processed_files.pkl")
        if os.path.exists(processed_path):
            with open(processed_path, "rb") as f:
                processed_files = pickle.load(f)

    # Scan PDFs in data folder
    pdf_files = [f for f in os.listdir("data") if f.endswith(".pdf")]
    new_docs = []

    for file in pdf_files:
        if file not in processed_files:
            loader = PyPDFLoader(os.path.join("data", file))
            new_docs.extend(loader.load())
            processed_files.add(file)

    # Split new documents into chunks
    if new_docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        new_chunks = splitter.split_documents(new_docs)
        if vectorstore:
            vectorstore.add_documents(new_chunks)
        else:
            vectorstore = FAISS.from_documents(new_chunks, embeddings)
        
        # Save index
        vectorstore.save_local("faiss_index")
        # Save processed files list
        processed_path = os.path.join("faiss_index", "processed_files.pkl")
        with open(processed_path, "wb") as f:
            pickle.dump(processed_files, f)

    # Initialize Groq LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=temp
    )

    # Build RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": k})
    )
    return qa_chain

# Initialize QA chain
qa_chain = load_chatbot(temperature, num_sources, groq_api_key)

# ------------------------------
# Initialize chat session state
# ------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ------------------------------
# Suggested questions
# ------------------------------
st.markdown("### 💡 Suggested Questions:")
col1, col2 = st.columns(2)
with col1:
    if st.button("📊 What is the expected value for Neopterin?"):
        st.session_state.messages.append({"role": "user", "content": "What is the expected value for Neopterin?"})
    if st.button("🔬 How do you measure Superoxide Dismutase?"):
        st.session_state.messages.append({"role": "user", "content": "How do you measure Superoxide Dismutase?"})
with col2:
    if st.button("⚡ What electrodes are used for Norepinephrine?"):
        st.session_state.messages.append({"role": "user", "content": "What electrodes are used for Norepinephrine?"})
    if st.button("📋 What tests are in the Kidney Panel?"):
        st.session_state.messages.append({"role": "user", "content": "What tests are in the Kidney Panel?"})

# ------------------------------
# User input
# ------------------------------
prompt = st.chat_input("Ask a question...")

# ------------------------------
# Process input and generate response
# ------------------------------
if prompt or st.session_state.messages:
    # Latest user message
    if prompt:
        user_message = prompt
        st.session_state.messages.append({"role": "user", "content": user_message})
    else:
        user_message = st.session_state.messages[-1]["content"]
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_message)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"query": user_message})
            response = result['result']
            st.write(response)
    
    # Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})