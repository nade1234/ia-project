

This project is an AI-powered **diet assistant** for French cuisine and nutrition.  
It allows you to query a collection of dietary plans and meals using natural language, powered by:

-  A vector database (Chroma)
-  OpenAI Embeddings and Chat Models
-  Markdown documents as data sources

## Features

Load French dietary plans from `.md` files  
 Create a local vector database (Chroma)  
 Query the database in natural language (French)  
Get practical and detailed responses  

---

## Project Structure

- `create_database.py` — Creates the vector database from your documents  
- `query_database.py` — Query the database with your questions  
- `data/books/*.md` — Your source Markdown documents  
- `chroma/` — The generated local vector database (auto-created)



