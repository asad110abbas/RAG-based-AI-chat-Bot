# KnowledgeBot AI Chat

KnowledgeBot is an AI-powered chatbot built using **LangChain**, HuggingFace embeddings, and OpenAI GPT-5-mini. It can answer questions from your own FAQ or document collection using retrieval-augmented generation (RAG) and provides a clean, interactive Gradio interface.

## Features

- Uses **LangChain** for document retrieval and prompt management.
- Supports custom FAQs in `.docx` format.
- Vector search with HuggingFace embeddings and Chroma.
- Generates responses using OpenAI GPT-5-mini.
- Interactive Gradio UI for easy chatting.

## Setup


```bash
git clone <repo-url>
cd <repo-name>

pip install -r requirements.txt
python app.py
export OPENAI_API_KEY="your_api_key_here"
