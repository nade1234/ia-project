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

# Configuration de l'environnement
load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"

PROMPT_TEMPLATE = """
Vous êtes un assistant diététicien IA expert spécialisé dans la cuisine française et la nutrition.

Répondez à la question en utilisant uniquement le contexte suivant, qui contient des informations sur les plans alimentaires et les repas :

{context}

---

Répondez à la question en vous basant sur le contexte ci-dessus en français : {question}

Fournissez une réponse détaillée et pratique avec des exemples spécifiques lorsque c'est possible.
"""

# ---------- Fonctions NLP ----------
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
    chunks = splitter.split_documents(documents)
    return [chunk for chunk in chunks if chunk.page_content.strip() and len(chunk.page_content) > 20]

def save_to_chroma(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    embedding = OpenAIEmbeddings()
    Chroma.from_documents(chunks, embedding, persist_directory=CHROMA_PATH)

def ensure_database():
    if not os.path.exists(CHROMA_PATH):
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
        return "❌ Aucun document .md trouvé dans le dossier `data/books`.", []

    try:
        embedding = OpenAIEmbeddings()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
        results = db.similarity_search_with_relevance_scores(query_text, k=5)

        if not results:
            return "❌ Aucun résultat trouvé pour votre question.", []

        filtered = [(doc, score) for doc, score in results if score >= 0.6]
        if not filtered:
            return "❌ Aucun résultat pertinent trouvé.", []

        context = "\n\n---\n\n".join([doc.page_content for doc, _ in filtered])
        sources = list(set(os.path.basename(doc.metadata.get("source", "inconnu")) for doc, _ in filtered))

        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        messages = prompt.format_messages(context=context, question=query_text)

        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
        response = model.invoke(messages)
        return response.content, sources

    except Exception as e:
        return f"❌ Erreur OpenAI : {e}", []

# ---------- Interface Streamlit ----------
st.set_page_config(page_title="Assistant Nutritionnel", page_icon="🥗")

# ---------- Style CSS ----------
st.markdown("""
<style>
    body {
        color: #000;
    }
    .response-container {
        background: #f8f9fa;
        color: #000 !important;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .example-questions {
        background: #f1f8e9;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #c8e6c9;
        color: #2e7d32;
        font-weight: 500;
    }
    .sources-container {
        background: #d0eaff;
        color: #000;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        font-size: 0.95rem;
    }
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e0e0e0;
        padding: 0.5rem 1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        color: white;
        font-weight: 500;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Titre ----------
st.markdown("""
<div class="header" style="text-align:center;padding:2rem;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);border-radius:15px;color:white;">
    <h1>🥗 Assistant Nutritionnel</h1>
    <p>Votre expert en nutrition personnalisé</p>
</div>
""", unsafe_allow_html=True)

# ---------- Saisie ----------
st.markdown("### 💬 Posez votre question")
col1, col2 = st.columns([2, 1])
with col1:
    query = st.text_input("", placeholder="Ex: Quel est un exemple de dîner équilibré ?", label_visibility="collapsed")
with col2:
    search_button = st.button("🔍 Rechercher")

# ---------- Exemples ----------
st.markdown("""
<div class="example-questions">
    💡 <strong>Exemples de questions :</strong><br><br>
    • Que manger pour le petit-déjeuner ?<br>
    • Quels aliments sont riches en fer ?<br>
    • Comment composer un repas équilibré ?<br>
    • Quelles sont les meilleures collations saines ?
</div>
""", unsafe_allow_html=True)

# ---------- Résultat ----------
if search_button and query.strip():
    with st.spinner("🤔 L'expert analyse votre question..."):
        response, sources = query_database(query.strip())
    
    st.markdown("### 💬 Réponse de l'expert :")
    st.markdown(f"<div class='response-container'>{response}</div>", unsafe_allow_html=True)

    if sources and not response.startswith("❌"):
        st.markdown("### 📚 Sources consultées :")
        st.markdown(f"""
        <div class='sources-container'>
            <strong>Fichiers consultés :</strong><br>
            {'<br>• '.join([''] + sources)}
        </div>
        """, unsafe_allow_html=True)

elif search_button and not query.strip():
    st.warning("⚠️ Veuillez saisir une question avant de rechercher.")

# ---------- Footer ----------
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666; padding: 1rem;'>🥗 Assistant Nutritionnel - Propulsé par l'IA</div>", unsafe_allow_html=True)
