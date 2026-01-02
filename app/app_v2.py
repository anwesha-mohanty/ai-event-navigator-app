import streamlit as st
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os
from datetime import datetime

# Load environment variables
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load FAISS index and metadata
index = faiss.read_index("vectorstore/events.index")
with open("vectorstore/meta.pkl", "rb") as f:
    metadata = pickle.load(f)

# Ensure metadata is a list of dicts
metadata = [r for r in metadata if isinstance(r, dict)]

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    return embed_model.encode(text, convert_to_numpy=True)

# Search FAISS
def search_index(query_vector, k=10):
    distances, indices = index.search(np.array([query_vector], dtype="float32"), k)
    results = []
    for idx in indices[0]:
        if idx < len(metadata):
            results.append(metadata[idx])
    return results

# Generate answer with Groq
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
        Classification: {r.get('classification', 'Other')}
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
        temperature=0.7,  # deterministic
    )
    return response.choices[0].message.content

# Streamlit UI
st.set_page_config(page_title="Event Navigator", page_icon="🎟️", layout="wide")
st.title("🎟️ AI Event Navigator")

# Sidebar filters
cities = sorted(list({r.get("city") or "Unknown" for r in metadata if isinstance(r, dict)}))
classifications = sorted(list({r.get("classification") or "Other" for r in metadata if isinstance(r, dict)}))

st.sidebar.header("Filters")
selected_city = st.sidebar.selectbox("City", ["All"] + cities)
start_date = st.sidebar.date_input("Start Date", datetime.today())
end_date = st.sidebar.date_input("End Date", datetime.today())
selected_class = st.sidebar.multiselect("Event Type", classifications, default=classifications)

# User query input
user_query = st.text_input("🔍 Describe the type of event you're looking for:")

if st.button("Find Events") and user_query:
    with st.spinner("Searching events..."):
        query_vector = get_embedding(user_query)
        results = search_index(query_vector, k=20)

        # Apply filters
        filtered_results = []
        for r in results:
            try:
                event_date = datetime.strptime(r["date"], "%Y-%m-%d")
            except:
                continue  # skip malformed dates
            if (selected_city != "All" and r.get("city") != selected_city):
                continue
            if not (start_date <= event_date <= end_date):
                continue
            if r.get("classification","Other") not in selected_class:
                continue
            filtered_results.append(r)

        # Limit top 5 after filtering
        filtered_results = filtered_results[:5]

        if not filtered_results:
            st.warning("No events matched your query and filters.")
        else:
            # Generate AI recommendation
            answer = generate_answer(user_query, filtered_results)

            st.subheader("✨ Recommendations")
            st.write(answer)

            st.subheader("📌 Matching Events")
            for r in filtered_results:
                with st.container():
                    st.markdown(f"### {r['name']}")
                    st.write(f"📍 **City:** {r.get('city','N/A')}")
                    st.write(f"🏟️ **Venue:** {r.get('venue','N/A')}")
                    st.write(f"📅 **Date:** {r.get('date','N/A')}")
                    st.write(f"🎭 **Type:** {r.get('classification','Other')}")
                    st.write(f"🔗 [More Info]({r.get('url','#')})")
                    st.write("---")
