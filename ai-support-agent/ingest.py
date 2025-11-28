# ingest.py
import os, pickle, sys
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm

MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 700
OVERLAP = 100
INDEX_DIR = "index"

def chunk_text(text):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        start += CHUNK_SIZE - OVERLAP
    return chunks

def ingest(files):
    os.makedirs(INDEX_DIR, exist_ok=True)
    model = SentenceTransformer(MODEL_NAME)
    docs = []
    embeddings = []

    for f in files:
        with open(f, "r", encoding="utf-8") as file:
            text = file.read()
        chunks = chunk_text(text)
        for i, ch in enumerate(chunks):
            doc_id = f"{os.path.basename(f)}_{i}"
            docs.append({"id": doc_id, "text": ch, "source": f})
            embeddings.append(model.encode(ch))

    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, f"{INDEX_DIR}/faiss.index")
    with open(f"{INDEX_DIR}/docs.pkl", "wb") as f:
        pickle.dump(docs, f)

    print("Indexing finished — total chunks:", len(docs))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py data/faq.txt")
        exit()

    ingest(sys.argv[1:])
