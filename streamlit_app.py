import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
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

# Streamlit UI
st.title("🔍 SHL Assessment Recommender")
st.markdown("Enter a job description or role to get assessment suggestions.")

user_input = st.text_area("✍️ Input Job Description or Role")

if st.button("🎯 Get Recommendations"):
    if user_input.strip() != "":
        results = get_top_n(user_input)
        st.success("Top Recommendations:")
        st.table(pd.DataFrame(results))
    else:
        st.warning("Please enter a valid input.")

