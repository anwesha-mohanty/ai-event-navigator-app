import os, pickle, faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import json

# Load GROQ API key from .env
from dotenv import load_dotenv
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- Load FAISS index + metadata ---
index = faiss.read_index("vectorstore/events.index")
with open("vectorstore/meta.pkl", "rb") as f:
    meta = pickle.load(f)

# --- Load embedding model ---
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def search_events(query, k=5):
    """Search events relevant to user query"""
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb.astype("float32"), k)
    
    results = []
    for idx in I[0]:
        if idx < len(meta):
            results.append(meta[idx])
    return results

def generate_answer(query, results):
    """Use LLM to summarize results into natural answer"""
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
    Mention event name, date, city, and price if available.
    Suggest which one fits best and why.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # or "gpt-4o" if available
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    print("🔍 Event Navigator — Search")
    while True:
        query = input("\nEnter your query (or 'exit'): ")
        if query.lower() == "exit":
            break
        results = search_events(query, k=5)
        if not results:
            print("No results found.")
            continue
        answer = generate_answer(query, results)

        #save results, answer to json file
        output = {
        "query": query,
        "answer": answer,
        "results": results
        }

        with open("query_output.json", "w") as f:
            json.dump(output, f, indent=2)

        print("\n✨ Recommendation:\n")
        print(answer)
