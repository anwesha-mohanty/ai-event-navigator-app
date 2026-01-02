import os, json, pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def chunk_text(text, chunk_size=80, overlap=20):
    """Split text into overlapping word chunks"""
    tokens = text.split()
    result=[]
    i=0
    while i < len(tokens):
        chunk = " ".join(tokens[i:i+chunk_size])
        result.append(chunk)
        i += chunk_size - overlap
    return result

if __name__ == "__main__":
    with open("data/events.json", "r") as f:
        events = json.load(f)

    docs = []
    for ev in events:
        chunks = chunk_text(ev["description"] or "")
        for i, ch in enumerate(chunks):
            docs.append({
                "id": f"{ev['id']}_{i}",
                "event_id": ev["id"],
                "name": ev["name"],
                "date": ev["date"],
                "venue": ev["venue"],
                "city": ev["city"],
                "chunk": ch,
                "url": ev["url"],
                "classification": ev.get("classification")
            })

    print(f"Total chunks: {len(docs)}")

    # Load embedding model (local)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [d["chunk"] for d in docs]
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # Create FAISS index
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs.astype("float32"))

    os.makedirs("vectorstore", exist_ok=True)
    faiss.write_index(index, "vectorstore/events.index")
    with open("vectorstore/meta.pkl", "wb") as f:
        pickle.dump(docs, f)

    print("✅ Built FAISS index and saved to vectorstore/")
