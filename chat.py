from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Load the vector database
print("Lodaing Database...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Use Ollma LLM for Q&A
llm = Ollama(model="llama3.2", temperature=0.7)

# Create Q&A chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# Chat koop 
print("\n Welcome to the UCB NanoTech Chatbot! Ask away. (type 'quit' to exit)\n")

while True:
    question = input("You:")
    
    if question.lower() in ['quit', 'exit', 'q']:
        break
    
    if question.strip():
        result = qa_chain.invoke({"query": question})
        print(f"\nBot: {result['result']}\n")