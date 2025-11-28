# ai_support_assistant
AI Support Assistant — Gemini RAG Chatbot

An intelligent support assistant powered by Google Gemini, FAISS, and Streamlit
Overview

The AI Support Assistant is a fully functional RAG-based chatbot built using:

Google Gemini Flash as the LLM

FAISS as the vector database

Sentence Transformers for embeddings

Streamlit for UI

CSV for ticket management

This assistant answers questions from your custom FAQ documents and creates support tickets when it cannot answer.

Perfect for:

✔ Startups
✔ College final year projects
✔ Internal support systems
✔ Customer helpdesk automation
| Component  | Technology            |
| ---------- | --------------------- |
| Language   | Python 3.10+          |
| LLM        | Google Gemini Flash   |
| Embeddings | Sentence Transformers |
| Vector DB  | FAISS                 |
| UI         | Streamlit             |
| Storage    | Local CSV             |
| Env Vars   | .env                  |

ai-support-agent/
│── app.py
│── ingest.py
│── requirements.txt
│── .env
│── data/
│     └── faq.txt
│── index/
│     ├── faiss.index
│     └── docs.pkl
└── tickets.csv

