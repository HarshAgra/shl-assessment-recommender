from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load data
df = pd.read_excel("shl_assessments.xlsx")
df = df.dropna(subset=['Assessment Name', 'Description'])
df = df.fillna('N/A')
data = df.to_dict(orient='records')

# Create embeddings
descriptions = [item['Description'] for item in data]
embeddings = model.encode(descriptions)

# Get top N similar assessments
def get_top_n(query, n=10):
    query_embedding = model.encode([query])
    sims = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:n]
    return [data[i] for i in top_indices]

# FastAPI app
app = FastAPI()

# Health check/root
@app.get("/")
def root():
    return {"message": "SHL Assessment Recommender API is live. Use POST /recommend"}

# Query schema
class Query(BaseModel):
    text: str

# Recommendation endpoint
@app.post("/recommend")
def recommend(query: Query):
    results = get_top_n(query.text)
    formatted_results = [
        {
            "Assessment Name": r["Assessment Name"],
            "URL": r.get("URL", "N/A"),
            "Remote Testing Support": r.get("Remote Testing Support", "N/A"),
            "Adaptive/IRT Support": r.get("Adaptive/IRT Support", "N/A"),
            "Duration": r.get("Duration", "N/A"),
            "Test Type": r.get("Test Type", "N/A"),
        }
        for r in results
    ]
    return formatted_results
