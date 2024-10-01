import re
import numpy
import pandas as pd
from fastapi import FastAPI
import ollama
from sentence_transformers import SentenceTransformer
import uvicorn
import faiss
import json


def remove_numbers(text: str):
    """Removes numbers from a string.

    Args:
      text(str): The string to remove numbers from.

    Returns:
      The string with numbers removed.
    """
    return re.sub(r'\d+', '', text)


index = faiss.read_index('embeddings.index')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # Loading sentence transformer used to embed the query
file_path = 'book_database.csv'  # Book database
with open(file_path, 'r', encoding='utf-8') as file:
    df = pd.read_csv(file)

df.fillna('Unknown', inplace=True)  # Fills any blank cells in the dataset with a value to prevent errors


# Step 2: Combine the 'description' and 'categories' columns for generating embeddings

df['combined_text'] = df['categories'] + " " + df['description']  # Concatenating description and category

# Extract relevant text from the new 'combined_text' column and other metadata
combined_texts = df['combined_text'].tolist()
titles = df['Title'].tolist()
authors = df['authors'].tolist()
categories = df['categories'].tolist()
year = df['published_year'].tolist()

# llama3.1:8b is the model being used

app = FastAPI()  # Initiate FastAPI


@app.get("/{genre}")
def read_genre(genre: str):
    query = f"Recommend me a {genre} book."  # User query
    query_embedding = model.encode([query])

    k = 9  # Number of books to retrieve based on L2 distance
    distances, indices = index.search(query_embedding, k)

    # Step 6: Get the relevant text and metadata from the retrieved results
    retrieved_data = []
    for i in indices[0]:
        retrieved_entry = {
            "title": titles[i],
            "authors": authors[i],
            "category": categories[i],
            "year": year[i],
            "description": combined_texts[i]  # This now contains both desc and cat

        }
        retrieved_data.append(retrieved_entry)
    context = "\n\n".join([
        f"Title: {item['title']}\nAuthors: {item['authors']}\nCategory: {item['category']}\nYear: {item['year']}\nDescription & Category: {item['description']} "
        for item in retrieved_data])  # Context generated based on RAG

    prompt = f'Based on the following information:\n{context}\n whilst also considering their preferred genre which is {genre}. Recommend exactly 3 books in JSON format precisely using this model answer. Comma separate the 3 Jsons.' \
             f'[{{"Book": "Title", "Author": "["Author1", "Author2"]", "Reason": "Reason for recommendation"}}'

    output = ollama.generate(model='llama3.1:8b', prompt=f'{prompt}')  # Generate response using the ollama model

    clean_output = output["response"].replace('[', '').replace(']', '').replace('}', '},').replace('},,', '},')  # Remove any square brackets in the response that have been added
    start_point = output["response"].find("{")  # Find beginning of the Json as model adds text to the start of response
    end_point = output["response"].rfind("}")  # Find end of the final Json as model adds text to the end of response
    book_recommendations = clean_output[start_point - 1:end_point + 1]  # sets output to only the
    book_recommendations = book_recommendations.replace('"Author":', '"Author":[')  #Add square brackets so list of authors is returned rather than comma separated author names
    book_recommendations = book_recommendations.replace(', "Reason', '], "Reason').replace(',"Reason', '], "Reason')  #Add square brackets so list of authors is returned rather than comma separated author names
    if book_recommendations[-1] == ',':  # Sometimes a comma is added to the end so this ignores the comma in the output
        book_recommendations = book_recommendations[:-1]

    book_recommendations = r'[' + book_recommendations + ']'  # Turns the output into a list so it can be taken in by json load
    return json.loads(book_recommendations)

