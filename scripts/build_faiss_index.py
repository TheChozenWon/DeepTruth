import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Specify the name of your CSV file here
CSV_FILENAME = "balanced_train_data.csv"

# Read the dataset directly from the current directory
df = pd.read_csv(CSV_FILENAME)

# Adjust the column name based on your dataset.
# Commonly used columns might be 'statement' or 'news_text'
if 'statement' in df.columns:
    texts = df['statement'].tolist()
elif 'news_text' in df.columns:
    texts = df['news_text'].tolist()
else:
    raise ValueError("Dataset must contain a 'statement' or 'news_text' column.")

# Initialize the all-MiniLM-L6-v2 model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
print("Encoding fact-check texts...")
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings, dtype='float32')

# Create a FAISS index using L2 distance
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save the index to the current directory
faiss.write_index(index, 'fact_checks.index')
print("FAISS index built and saved as fact_checks.index")
