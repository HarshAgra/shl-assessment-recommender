import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load data
df = pd.read_excel("shl_assessments.xlsx")
df.columns = df.columns.str.strip()
df = df.dropna(subset=['Assessment Name', 'Description'])
df = df.fillna('N/A')
data = df.to_dict(orient='records')

# Extract duration in minutes from query
def extract_duration(query):
    match = re.search(r'(\d+)\s*minutes?', query.lower())
    return int(match.group(1)) if match else None

# Get top N similar assessments with optional duration filter
def get_top_n(query, n=10):
    duration_limit = extract_duration(query)

    # Filter by duration if limit found
    filtered_data = [
        item for item in data
        if (
            item.get("Duration", "N/A") != "N/A" and
            re.search(r'\d+', str(item["Duration"])) and
            int(re.search(r'\d+', str(item["Duration"])).group()) <= duration_limit
        )
    ] if duration_limit else data

    # Encode descriptions
    descriptions = [item["Description"] for item in filtered_data]
    embeddings = model.encode(descriptions)

    # Compute similarity
    query_embedding = model.encode([query])
    sims = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:n]

    # Prepare results
    top_results = []
    for idx, i in enumerate(top_indices, 1):
        item = filtered_data[i]
        url = item.get('URL', '#')
        link = f'<a href="{url}" target="_blank">Link</a>'  # Visible as "Link"
        top_results.append({
            "Assessment": idx,
            "Assessment Name": item.get("Assessment Name", "N/A"),
            "URL": link,
            "Remote Testing Support": item.get("Remote Testing Support", "N/A"),
            "Adaptive/IRT Support": item.get("Adaptive/IRT Support", "N/A"),
            "Duration": item.get("Duration", "N/A"),
            "Test Type": item.get("Test Type", "N/A"),
        })

    return top_results

# Streamlit UI
st.title("🔍 SHL Assessment Recommender")
st.markdown("Enter a job description or role to get assessment suggestions. You can also mention a duration like '30 minutes'.")

user_input = st.text_area("✍️ Input Job Description or Role")

if st.button("🎯 Get Recommendations"):
    if user_input.strip():
        results = get_top_n(user_input)
        st.success("Top Recommendations:")
        df_result = pd.DataFrame(results)
        st.markdown(df_result.to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.warning("Please enter a valid input.")



