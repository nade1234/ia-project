# -*- coding: utf-8 -*-
import os
import sys
import shutil
import glob

# Forcer l'encodage UTF-8 pour Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import nltk

# Télécharger les ressources NLTK si nécessaire
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Charger les variables d'environnement
load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"

def main():
    print(" Starting database creation...")
    generate_data_store()

def generate_data_store():
    """Generate the vector store from documents"""
    try:
        documents = load_documents()
        if not documents:
            print(" No documents found. Check your data directory.")
            return
            
        chunks = split_text(documents)
        if not chunks:
            print(" No valid chunks created. Check your documents.")
            return
            
        save_to_chroma(chunks)
        print(" Database creation completed successfully!")
        
    except Exception as e:
        print(f" Error during database creation: {e}")

def load_documents():
    """Load all markdown documents from the data directory"""
    documents = []
    
    if not os.path.exists(DATA_PATH):
        print(f" Data directory not found: {DATA_PATH}")
        return documents
    
    # Chercher tous les fichiers .md
    pattern = os.path.join(DATA_PATH, "*.md")
    files = glob.glob(pattern)
    
    print(f" Found {len(files)} markdown files in {DATA_PATH}")
    
    for filepath in files:
        try:
            print(f" Loading: {os.path.basename(filepath)}")
            
            # Essayer différents encodages
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1']:
                try:
                    loader = TextLoader(filepath, encoding=encoding)
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"    Loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                print(f"    Could not load {filepath} - encoding issues")
                
        except Exception as e:
            print(f"    Error loading {filepath}: {e}")
    
    print(f" Total documents loaded: {len(documents)}")
    return documents

def split_text(documents: list[Document]):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,  # Augmenté pour plus de contexte
        chunk_overlap=50,  # Réduit pour éviter trop de redondance
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""]  # Séparateurs plus intelligents
    )
    
    print(" Splitting documents into chunks...")
    chunks = text_splitter.split_documents(documents)
    
    # Filtrer les chunks vides et trop courts
    valid_chunks = []
    for chunk in chunks:
        content = chunk.page_content.strip()
        if content and len(content) > 20:  # Au moins 20 caractères
            valid_chunks.append(chunk)
        else:
            print(f" Skipping short/empty chunk from {chunk.metadata.get('source', 'unknown')}")
    
    print(f"Split {len(documents)} documents into {len(valid_chunks)} valid chunks")
    print(f"   (removed {len(chunks) - len(valid_chunks)} invalid chunks)")
    
    if valid_chunks:
        # Afficher un exemple de chunk
        sample_chunk = valid_chunks[0]
        print("\n Sample chunk preview:")
        print("-" * 50)
        print(sample_chunk.page_content[:200] + "..." if len(sample_chunk.page_content) > 200 else sample_chunk.page_content)
        print("-" * 50)
        print(f"Metadata: {sample_chunk.metadata}")
        print("-" * 50)
    
    return valid_chunks

def save_to_chroma(chunks: list[Document]):
    """Save chunks to Chroma vector database"""
    print(f" Saving {len(chunks)} chunks to vector database...")
    
    # Supprimer l'ancienne base de données
    if os.path.exists(CHROMA_PATH):
        print(f" Removing existing database at {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)

    try:
        # Créer la nouvelle base de données
        print(" Creating embeddings...")
        embedding_function = OpenAIEmbeddings()
        
        print(" Creating Chroma database...")
        db = Chroma.from_documents(
            chunks, 
            embedding_function, 
            persist_directory=CHROMA_PATH
        )
        
        # Note: persist() n'est plus nécessaire avec les nouvelles versions
        print(f" Successfully saved {len(chunks)} chunks to {CHROMA_PATH}")
        
        # Test rapide de la base
        print(" Testing database...")
        test_results = db.similarity_search("dîner", k=1)
        if test_results:
            print(" Database test successful")
        else:
            print(" Database test returned no results")
            
    except Exception as e:
        print(f" Error saving to Chroma: {e}")
        raise

if __name__ == "__main__":
    main()