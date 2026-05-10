# UCB NanoTech RAG Chatbot

A retrieval-augmented generation (RAG) chatbot for querying UC Berkeley nanotechnology research documents. Upload PDFs and ask questions about the content — the chatbot finds the relevant sections and answers using an LLM.

## Features

- Ingests and indexes PDF research documents using FAISS vector search
- Answers questions grounded in the documents (no hallucination of sources)
- Streamlit web UI with suggested questions and adjustable settings
- Terminal chat interface for quick queries
- Powered by Groq (LLaMA 3.3 70B) and HuggingFace embeddings

## Project Structure

```
rag-chatbot/
├── streamlit_app.py   # web UI
├── chat.py            # terminal chat interface
├── ingest.py          # builds the FAISS index from PDFs
├── requirements.txt
└── data/              # place your PDF files here (not tracked by git)
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Add a `.env` file with your Groq API key:
```
GROQ_API_KEY=your_key_here
```

3. Add PDF files to the `data/` folder, then build the index:
```bash
python ingest.py
```

## Usage

**Web UI:**
```bash
streamlit run streamlit_app.py
```

**Terminal:**
```bash
python chat.py
```

## Getting a Groq API Key

Sign up at [console.groq.com](https://console.groq.com) — it's free.
