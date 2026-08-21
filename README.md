# RAG Company Knowledge Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that retrieves relevant information from company documents and uses an AI model to generate grounded responses.

##  Project Overview

This project is being built to understand and implement a complete RAG pipeline from document ingestion to intelligent question answering.

The chatbot will allow users to ask questions about a collection of company-related documents and receive answers based on the retrieved information.

##  Project Architecture

User
↓
Streamlit Frontend
↓
FastAPI Backend
↓
Document Processing
↓
Document Retrieval
↓
Relevant Context
↓
LLM
↓
Grounded Answer

## 🛠️ Technologies Used

- Python
- FastAPI
- Streamlit
- LangChain
- RAG
- Large Language Models
- Git & GitHub

## 📂 Project Structure

```text
backend/
├── main.py
└── ingest.py

frontend/
└── app.py

data/
└── Sample knowledge-base documents
