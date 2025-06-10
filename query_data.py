# -*- coding: utf-8 -*-
import os
import sys
import argparse

if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a professional AI dietitian specialized in healthy eating and nutrition.

Use only the following context, which contains information about dietary plans and meals:

{context}

---

Answer the question in English based on the above context: {question}

Give a detailed and practical response with specific examples when possible.
"""

def main():
    parser = argparse.ArgumentParser(description="Query your nutritional database.")
    parser.add_argument("query_text", type=str, help="The question you want to ask.")
    args = parser.parse_args()
    query_text = args.query_text

    if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
        print(f"❌ No vector store found in '{CHROMA_PATH}'. Please run the Streamlit app at least once to generate it.")
        return

    try:
        embedding_function = OpenAIEmbeddings()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        print("✅ Database loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading database: {e}")
        return

    print(f"\n🔍 Searching for relevant context for: \"{query_text}\"\n")

    try:
        results = db.similarity_search_with_relevance_scores(query_text, k=5)
        print(f"Found {len(results)} potential results")
    except Exception as e:
        print(f"❌ Error during search: {e}")
        return

    if len(results) == 0:
        print(f"❌ No matching results found for: '{query_text}'")
        return

    filtered_results = [(doc, score) for doc, score in results if score >= 0.6]

    if len(filtered_results) == 0:
        print(f"❌ No results with relevance score >= 0.6.")
        return

    print(f"✅ Found {len(filtered_results)} relevant results")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in filtered_results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    print("\n🧠 Generating response...")

    try:
        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, max_tokens=500)
        response = model.invoke(prompt)
        response_text = response.content
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return

    sources = [doc.metadata.get("source", "unknown") for doc, _ in filtered_results]

    print("\n" + "="*60)
    print("ANSWER:")
    print("="*60)
    print(response_text)
    print("\n" + "="*60)
    print("SOURCES:")
    print("="*60)
    for i, source in enumerate(set(sources), 1):
        print(f"{i}. {source}")
    print("="*60)

if __name__ == "__main__":
    main()
