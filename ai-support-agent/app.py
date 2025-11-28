# app.py
import os
import pickle
import streamlit as st
from dotenv import load_dotenv
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import csv
import datetime
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "models/gemini-flash-latest"
EMB_MODEL = "all-MiniLM-L6-v2"
INDEX_DIR = "index"

@st.cache_resource
def load_components():
    model = SentenceTransformer(EMB_MODEL)
    index = faiss.read_index(f"{INDEX_DIR}/faiss.index")
    with open(f"{INDEX_DIR}/docs.pkl", "rb") as f:
        docs = pickle.load(f)
    return model, index, docs

def search_docs(query, k=4):
    model, index, docs = load_components()
    q_emb = model.encode([query]).astype("float32")
    D, I = index.search(q_emb, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < len(docs):
            item = docs[idx]
            item["score"] = float(score)
            results.append(item)
    return results

def call_llm(prompt):
    model = genai.GenerativeModel(MODEL)
    response = model.generate_content(prompt)
    return response.text


def save_ticket(user_text):
    ticket = {
        "id": datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
        "message": user_text,
        "status": "Open"
    }
    file_exists = os.path.isfile("tickets.csv")
    with open("tickets.csv", "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ticket.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(ticket)
    return ticket["id"]

st.title("🧠 AI Support Assistant (Gemini 1.5 Flash)")
st.write("Ask any question based on your company FAQ documents.")

if "history" not in st.session_state:
    st.session_state["history"] = []

query = st.chat_input("Type your question here...")

if query:
    st.session_state.history.append(("user", query))

    with st.spinner("Searching knowledge base..."):
        docs = search_docs(query, k=4)

    context = "\n\n".join([f"[{d['id']}]: {d['text']}" for d in docs])

    prompt = f"""
Use ONLY the following context to answer the user's question.
If the answer is not available, say "I don't know. Do you want me to create a support ticket?"

CONTEXT:
{context}

USER QUESTION:
{query}
"""

    with st.spinner("Generating response..."):
        answer = call_llm(prompt)

    st.session_state.history.append(("assistant", answer))

for role, msg in st.session_state.history:
    if role == "user":
        st.write(f"**🧑 You:** {msg}")
    else:
        st.write(f"**🤖 Assistant:** {msg}")

st.markdown("---")
st.subheader("Create Ticket")

if st.button("Create Support Ticket"):
    last_user_msg = ""
    for role, msg in reversed(st.session_state.history):
        if role == "user":
            last_user_msg = msg
            break

    ticket_id = save_ticket(last_user_msg)
    st.success(f"Ticket created successfully! Ticket ID: {ticket_id}")
