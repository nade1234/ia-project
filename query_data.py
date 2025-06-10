# -*- coding: utf-8 -*-
import os
import sys
import argparse

# Forcer l'encodage UTF-8 pour Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are an expert dietitian AI assistant specialized in French cuisine and nutrition.

Answer the question using only the following context, which contains information about dietary plans and meals:

{context}

---

Answer the question based on the above context in French: {question}

Provide a detailed and practical response with specific examples when possible.
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser(description="Query your dietary plan database.")
    parser.add_argument("query_text", type=str, help="The query text (your question).")
    args = parser.parse_args()
    query_text = args.query_text

    # Check if database exists
    if not os.path.exists(CHROMA_PATH):
        print(f" Database not found at {CHROMA_PATH}. Please run create_database.py first.")
        return

    # Prepare the DB.
    try:
        embedding_function = OpenAIEmbeddings()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        print(" Database loaded successfully")
    except Exception as e:
        print(f" Error loading database: {e}")
        return

    # Search the DB.
    print(f"\n🔍 Searching for relevant context to answer: \"{query_text}\"\n")
        
    try:
        results = db.similarity_search_with_relevance_scores(query_text, k=5)
        print(f"Found {len(results)} potential results")
    except Exception as e:
        print(f" Error during search: {e}")
        print("This usually means the database is corrupted. Try recreating it with create_database.py")
        return

    if len(results) == 0:
        print(f" Unable to find any matching results for: '{query_text}'")
        return

    # Filter out results with low relevance score (< 0.6 for more flexibility).
    filtered_results = [(doc, score) for doc, score in results if score >= 0.6]

    if len(filtered_results) == 0:
        print(f" No results with sufficient relevance score found (threshold: 0.6).")
        print("\n📋 Available results with lower scores:")
        for i, (doc, score) in enumerate(results[:3], 1):
            print(f"  {i}. Score: {score:.3f} - {doc.page_content[:150]}...")
        return

    print(f" Found {len(filtered_results)} relevant results")
    
    # Show relevance scores
    for i, (doc, score) in enumerate(filtered_results, 1):
        print(f"  Result {i}: Score {score:.3f}")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in filtered_results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    print("\n Generating response...")
    
    try:
        model = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.2,
            max_tokens=500
        )
        
        # Utiliser invoke() au lieu de predict()
        response = model.invoke(prompt)
        response_text = response.content
        
    except Exception as e:
        print(f" Error generating response: {e}")
        return

    sources = [doc.metadata.get("source", "unknown") for doc, _score in filtered_results]
    
    print("\n" + "="*60)
    print("RÉPONSE:")
    print("="*60)
    print(response_text)
    print("\n" + "="*60)
    print(" SOURCES:")
    print("="*60)
    for i, source in enumerate(set(sources), 1):
        print(f"{i}. {source}")
    print("="*60)

if __name__ == "__main__":
    main()