from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the FAISS index from the current directory
index = faiss.read_index('fact_checks.index')

# Initialize the same model used to build the index
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define an example query (replace this with any text you want to verify)
query = "Example statement to verify the facts."
query_embedding = model.encode([query], show_progress_bar=False)
query_embedding = np.array(query_embedding, dtype='float32')

# Query the index for the top 5 most similar fact-checks
k = 5
distances, indices = index.search(query_embedding, k)
print("Top 5 similar fact-check indices:", indices)
print("Corresponding distances:", distances)
