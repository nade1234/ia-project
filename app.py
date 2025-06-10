# -*- coding: utf-8 -*-
import streamlit as st
import os
import shutil
import glob
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"

PROMPT_TEMPLATE = """
You are a professional AI dietitian specialized in healthy eating and nutrition.

Use only the following context, which contains information about dietary plans and meals:

{context}

---

Answer the question in English based on the above context: {question}

Give a detailed and practical response with specific examples when possible.
"""

# --------- Helper Functions ---------

def load_documents():
    documents = []
    if not os.path.exists(DATA_PATH):
        return documents

    pattern = os.path.join(DATA_PATH, "*.md")
    files = glob.glob(pattern)

    for filepath in files:
        for encoding in ['utf-8', 'utf-8-sig', 'latin-1']:
            try:
                loader = TextLoader(filepath, encoding=encoding)
                docs = loader.load()
                documents.extend(docs)
                break
            except UnicodeDecodeError:
                continue
    return documents

def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    return [chunk for chunk in splitter.split_documents(documents) if chunk.page_content.strip() and len(chunk.page_content) > 20]

def save_to_chroma(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    embedding = OpenAIEmbeddings()
    Chroma.from_documents(chunks, embedding, persist_directory=CHROMA_PATH)

def ensure_database():
    if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
        docs = load_documents()
        if not docs:
            return False
        chunks = split_text(docs)
        if not chunks:
            return False
        save_to_chroma(chunks)
    return True

def query_database(query_text):
    if not ensure_database():
        return "❌ No markdown files found in the `data/books` directory.", []

    try:
        embedding = OpenAIEmbeddings()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
        results = db.similarity_search_with_relevance_scores(query_text, k=5)

        if not results:
            return "❌ No results found for your question.", []

        filtered = [(doc, score) for doc, score in results if score >= 0.6]
        if not filtered:
            return "❌ No relevant results found.", []

        context = "\n\n---\n\n".join([doc.page_content for doc, _ in filtered])
        sources = list(set(os.path.basename(doc.metadata.get("source", "unknown")) for doc, _ in filtered))

        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        messages = prompt.format_messages(context=context, question=query_text)

        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
        response = model.invoke(messages)
        return response.content, sources

    except Exception as e:
        return f"❌ Error: {e}", []

# --------- Streamlit UI ---------

st.set_page_config(page_title="Nutritional Assistant", page_icon="🥗")

# ---- Header ----
st.markdown("""
<div class="header" style="text-align:center;padding:2rem;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);border-radius:15px;color:white;">
    <h1>🥗 Nutritional Assistant</h1>
    <p>Your personalized AI nutrition expert</p>
</div>
""", unsafe_allow_html=True)

# ---- Input ----
st.markdown("### 💬 Ask your question")
col1, col2 = st.columns([2, 1])
with col1:
    query = st.text_input("", placeholder="e.g. What is a balanced dinner?", label_visibility="collapsed")
with col2:
    search_button = st.button("🔍 Search")

# ---- Example Questions ----
st.markdown("""
<div class="example-questions" style="background:#f1f8e9;padding:1rem;border-radius:8px;margin:1rem 0;border:1px solid #c8e6c9;color:#2e7d32;font-weight:500;">
    💡 <strong>Example questions:</strong><br><br>
    • What should I eat for breakfast?<br>
    • Which foods are rich in iron?<br>
    • How to create a balanced meal?<br>
    • What are some healthy snacks?
</div>
""", unsafe_allow_html=True)

# ---- Result Display ----
if search_button and query.strip():
    with st.spinner("🤔 The assistant is analyzing your question..."):
        response, sources = query_database(query.strip())

    st.markdown("### 💬 Expert's Response:")
    st.markdown(f"<div class='response-container' style='background:#f8f9fa;color:#000;padding:1.5rem;border-radius:10px;border-left:4px solid #667eea;margin:1rem 0;box-shadow:0 2px 4px rgba(0,0,0,0.1);'>{response}</div>", unsafe_allow_html=True)

    if sources and not response.startswith("❌"):
        st.markdown("### 📚 Sources used:")
        st.markdown(f"""
        <div class='sources-container' style='background:#d0eaff;color:#000;padding:1rem;border-radius:8px;margin-top:1rem;font-size:0.95rem;'>
            <strong>Files consulted:</strong><br>
            {'<br>• '.join([''] + sources)}
        </div>
        """, unsafe_allow_html=True)

elif search_button and not query.strip():
    st.warning("⚠️ Please enter a question before searching.")

# ---- Footer ----
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666; padding: 1rem;'>🥗 Nutritional Assistant - Powered by AI</div>", unsafe_allow_html=True)
