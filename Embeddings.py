import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss


file_path = 'book_database.csv'  # Replace with your file path
with open(file_path, 'r', encoding='utf-8') as file:
    df = pd.read_csv(file)

df.fillna('Unknown', inplace=True)
# Step 2: Combine the 'desc' and 'cat' columns for generating embeddings

df['combined_text'] = df['categories'] + " " + df['description']  # Concatenating description and category

# Extract relevant text from the new 'combined_text' column and other metadata
combined_texts = df['combined_text'].tolist()
titles = df['Title'].tolist()
authors = df['authors'].tolist()
categories = df['categories'].tolist()
year = df['published_year'].tolist()

# Step 3: Generate embeddings using TF-IDF (or any embedding model)
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(combined_texts)

# Step 4: Initialize FAISS index for similarity search
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, 'embeddings.index')  # Save the embeddings for use
