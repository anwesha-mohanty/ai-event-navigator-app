# 🎟️ AI Event Navigator

## Project Overview

AI Event Navigator is an AI-powered event discovery and recommendation system that allows users to find relevant concerts, sports, and live events using natural language queries.

The system combines **semantic vector search** with **large language models (LLMs)** to deliver personalized, context-aware recommendations. Instead of relying on keyword matching, event descriptions are embedded into a vector space and retrieved based on semantic similarity. A large language model then reasons over the retrieved results to generate clear, user-friendly recommendations.

This project demonstrates a practical **Retrieval-Augmented Generation (RAG)** pipeline, covering data ingestion, embedding, vector search, and LLM-based response generation.

**Core capabilities include:**
- Natural language event search (e.g., “live music this weekend in NYC”)
- Semantic retrieval using sentence embeddings and FAISS
- LLM-powered recommendation summaries using Groq (LLaMA)
- Interactive Streamlit-based user interface

---

## Setup & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ai-event-navigator-app.git
cd ai-event-navigator-app
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
Create a .env file in the project root with the following
```bash
TICKETMASTER_API_KEY=your_ticketmaster_api_key
GROQ_API_KEY=your_groq_api_key
```

### 4. Run the data pipeline
```bash
python ingest.py
python build_embeddings.py
```

### 5. Launch the application
```bash
streamlit run app.py
```

## Example Query

“Looking for live music events this weekend in New York”

## System Output:

Curated list of relevant events

Natural language explanation of why each event was recommended

Direct links for further details
