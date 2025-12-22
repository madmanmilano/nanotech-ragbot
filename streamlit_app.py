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
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    h1 {
        color: #1f77b4;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    .subtitle {
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/University_of_California%2C_Berkeley_logo.svg/1200px-University_of_California%2C_Berkeley_logo.svg.png", width=100)
    st.title("📚 About")
    st.markdown("""
    This AI-powered research assistant helps you explore UCB NanoTech lab documents using natural language queries.
    
    **How it works:**
    1. Ask any question about the research
    2. The AI searches through lab documents
    3. Get accurate answers with context
    
    **Example questions:**
    - What is the expected value for Neopterin?
    - How do you measure Superoxide Dismutase?
    - What electrodes are used for Norepinephrine?
    """)
    
    st.divider()
    
    st.markdown("### ⚙️ Settings")
    temperature = st.slider("Response Creativity", 0.0, 1.0, 0.7, 0.1)
    num_sources = st.slider("Number of Sources", 1, 5, 3)
    
    st.divider()
    
    st.markdown("### 📊 Stats")
    if "messages" in st.session_state:
        st.metric("Questions Asked", len([m for m in st.session_state.messages if m["role"] == "user"]))

# Main content - Set up page
st.title("🔬 UCB NanoTech Research Assistant")
st.markdown('<p class="subtitle">Ask questions about biomarker detection, electrochemical methods, and lab protocols</p>', unsafe_allow_html=True)

# Info boxes
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="info-box">
        <h4>🎯 Accurate</h4>
        <p>Answers based on actual lab documents</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="info-box">
        <h4>⚡ Fast</h4>
        <p>Get answers in seconds</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="info-box">
        <h4>🔒 Reliable</h4>
        <p>Powered by Groq & LangChain</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Load resources (cached so it only loads once)
@st.cache_resource
def load_chatbot(temp, k):
    with st.spinner("🔄 Loading research documents..."):
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
            temperature=temp
        )
        
        # Create Q&A chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
            return_source_documents=True
        )
        return qa_chain

qa_chain = load_chatbot(temperature, num_sources)

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍🔬" if message["role"] == "user" else "🤖"):
        st.write(message["content"])

# Suggested questions (only show if no messages yet)
if len(st.session_state.messages) == 0:
    st.markdown("### 💡 Try asking:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 What is the expected value for Neopterin?"):
            st.session_state.temp_question = "What is the expected value for Neopterin?"
            st.rerun()
        if st.button("🔬 How do you measure Superoxide Dismutase?"):
            st.session_state.temp_question = "How do you measure Superoxide Dismutase?"
            st.rerun()
    with col2:
        if st.button("⚡ What electrodes are used for Norepinephrine?"):
            st.session_state.temp_question = "What electrodes are used for Norepinephrine?"
            st.rerun()
        if st.button("📋 What tests are in the Kidney Panel?"):
            st.session_state.temp_question = "What tests are in the Kidney Panel?"
            st.rerun()

# Handle suggested question
if "temp_question" in st.session_state:
    prompt = st.session_state.temp_question
    del st.session_state.temp_question
else:
    # User input
    prompt = st.chat_input("💬 Ask a question about the research...")

# Process user input
if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍🔬"):
        st.write(prompt)
    
    # Get bot response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Searching research documents..."):
            result = qa_chain.invoke({"query": prompt})
            response = result['result']
            st.write(response)
            
            # Show sources (optional)
            if 'source_documents' in result and result['source_documents']:
                with st.expander("📚 View Sources"):
                    for i, doc in enumerate(result['source_documents'][:3]):
                        st.markdown(f"**Source {i+1}:**")
                        st.text(doc.page_content[:300] + "...")
                        st.divider()
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Built with ❤️ using Streamlit, LangChain & Groq | UCB NanoTech Lab</p>
</div>
""", unsafe_allow_html=True)