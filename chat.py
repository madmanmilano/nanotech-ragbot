from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()

print("Loading database...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

print("\nWelcome to the UCB NanoTech Chatbot! Ask away. (type 'quit' to exit)\n")

while True:
    question = input("You: ")

    if question.lower() in ['quit', 'exit', 'q']:
        break

    if question.strip():
        result = qa_chain.invoke({"query": question})
        print(f"\nBot: {result['result']}\n")