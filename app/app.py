import streamlit as st
from groq import Groq
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load FAISS index and metadata
import faiss, pickle

index = faiss.read_index("vectorstore/events.index")

with open("vectorstore/meta.pkl", "rb") as f:
    metadata = pickle.load(f)


# Helper: Search FAISS
def search_index(query_vector, k=5):
    distances, indices = index.search(np.array([query_vector], dtype="float32"), k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            results.append(metadata[idx])
    return results


from sentence_transformers import SentenceTransformer

# Load a local embedding model (runs on CPU/GPU)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    return embedder.encode(text).tolist()


# Helper: Generate answer with Groq
def generate_answer(query, results):
    context = ""
    for r in results:
        context += f"""
        Event: {r['name']}
        Date: {r['date']}
        City: {r['city']}
        Venue: {r['venue']}
        Info: {r['chunk']}
        URL: {r['url']}
        Classification: {r.get('classification')}
        ---
        """

    prompt = f"""
    You are an event recommendation assistant.
    User query: {query}
    Here are some event options:
    {context}

    Please recommend the top events in a clear, friendly way.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    return response.choices[0].message.content

# Streamlit UI
st.set_page_config(page_title="Event Navigator", page_icon="🎟️", layout="centered")
st.title("🎟️ AI Event Navigator")
st.write("Ask me about upcoming concerts, sports, and shows — powered by FAISS + Groq LLaMA.")

# Input
user_query = st.text_input("🔍 What kind of event are you looking for?")

if st.button("Find Events") and user_query:
    with st.spinner("Searching for the best events..."):
        # 1. Embed query
        query_vector = get_embedding(user_query)

        # 2. Search FAISS
        results = search_index(query_vector, k=5)

        # 3. Generate Answer
        answer = generate_answer(user_query, results)

    st.subheader("✨ Recommendations")
    st.write(answer)

    st.subheader("📌 Matching Events")
    for r in results:
        with st.expander(r["name"]):
            st.write(f"📍 **City:** {r['city']}")
            st.write(f"🏟️ **Venue:** {r['venue']}")
            st.write(f"📅 **Date:** {r['date']}")
            st.write(f"🎭 **Type:** {r.get('classification')}")
            st.write(f"🔗 [More Info]({r['url']})")
