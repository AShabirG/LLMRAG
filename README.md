This repository hosts the code for Bookworm AI, a book recommendation engine powered by a Large Language Model (LLM) and a Retrieval Augmented Generation (RAG) pipeline.

Bookworm AI takes a user's preferred genre as input and provides a curated list of book recommendations, along with concise explanations justifying each suggestion. The system leverages a vast knowledge base of books and their summaries to deliver accurate and insightful recommendations.

Features
Personalised Recommendations: Tailored book suggestions based on individual user preferences.
Explainable AI (XAI): Provides clear explanations for each recommendation, enhancing transparency and user trust.
FastAPI Integration: Delivers recommendations in JSON format through an API built using FastAPI.


Data Ingestion & Processing: A dataset of books, summaries, and relevant metadata is preprocessed and stored.
Knowledge Base: The processed data is indexed and stored in a vector database, forming the knowledge base for the RAG pipeline.
LLM & RAG Pipeline:
User input (preferred genre) is used to query the knowledge base.
The LLM, augmented by retrieved relevant information, generates book recommendations and explanations.
FastAPI: Exposes the functionality through a user-friendly API for seamless integration.




Run the FastAPI server:
uvicorn main:app --reload
API Endpoints
/{genre}:

Response: JSON array of book recommendations with author and explanation why it is a good fit.


Future Enhancements
More personalisation of output by taking in users response to the recommended books
Quantisation to increase speed of model
Feedback Mechanism: Allow users to provide feedback on recommendations to improve the system's accuracy.
This project demonstrates the power of LLMs and RAG pipelines in building intelligent recommendation systems. Feel free to explore the code!
