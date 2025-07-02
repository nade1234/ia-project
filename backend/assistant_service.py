import os

# ─── 1) Drop any inherited proxy env-vars ───
for proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(proxy_var, None)

# ─── 2) Monkey-patch ChatOpenAI before it’s imported elsewhere ───
import langchain_openai.chat_models
orig_init = langchain_openai.chat_models.ChatOpenAI.__init__
def patched_init(self, *args, **kwargs):
    kwargs.pop("proxies", None)
    orig_init(self, *args, **kwargs)
langchain_openai.chat_models.ChatOpenAI.__init__ = patched_init
print("🚀 Patched ChatOpenAI to remove 'proxies' kwarg")  # debug

# ─── 3) Now import everything else, including ChatOpenAI ───
import shutil
import glob
import re
from datetime import datetime
from urllib.parse import quote_plus
from typing import Literal, List, Optional 
from dotenv import load_dotenv

from pymongo import MongoClient
from pydantic import BaseModel

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# <-- Critical imports happen here -->
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

# (rest of your file unchanged…)



# Load environment variables
load_dotenv()

# Constants
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma")
DATA_PATH = os.getenv("DATA_PATH", "data/books")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
USE_LLM_FALLBACK = True

# MongoDB Setup

def initialize_mongodb():
    try:
        username = os.getenv("MONGO_USER")
        password = quote_plus(os.getenv("MONGO_PASS"))
        host = os.getenv("MONGO_HOST")
        uri = f"mongodb+srv://{username}:{password}@{host}/?retryWrites=true&w=majority"
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        db = client[os.getenv("MONGO_DB", "nutritional_assistant")]
        user_collection = db[os.getenv("MONGO_COLLECTION", "user_info_records")]
        client.server_info()
        return client, user_collection, True
    except Exception as e:
        print(f"MongoDB init error: {e}")
        return None, None, False

client, user_collection, mongo_status = initialize_mongodb()

# Pydantic Models
class UserInfo(BaseModel):
    name: str
    weight: float
    job: str
    sport: str

class ChatMessage(BaseModel):
    type: Literal["human", "ai"]
    content: str

# ------------ Utility Functions ------------

def extract_name_from_text(text: str) -> Optional[str]:
    prompt = PromptTemplate.from_template(
        "Extract only the person's name from this text. Return just the name, nothing else:\n\n{text}"
    )
    model = ChatOpenAI(model=MODEL_NAME, temperature=0)
    chain = prompt | model
    try:
        result = chain.invoke({"text": text})
        return result.content.strip()
    except:
        return None


def ask_gpt(prompt: str) -> str:
    model = ChatOpenAI(model=MODEL_NAME, temperature=0)
    try:
        resp = model.invoke(prompt)
        return resp.content.strip()
    except:
        return "no"


def detect_weight_in_text(text: str) -> bool:
    t = text.lower()
    patterns = [r"\d+\s*kg", r"\d+\s*kilo", r"weigh\s+\d+", r"weight\s+is\s+\d+", r"i\s+am\s+\d+\s*kg"]
    for p in patterns:
        if re.search(p, t): return True
    nums = re.findall(r"\b(\d+)\b", t)
    for num in nums:
        if 30 <= int(num) <= 200:
            idx = t.find(num)
            window = t[max(0, idx-20):idx+20]
            if any(w in window for w in ['kg','kilo','weight','weigh']): return True
    return False


def detect_job_in_text(text: str) -> bool:
    t = text.lower()
    keywords = ['work','job','profession','engineer','teacher','developer','manager','nurse','doctor']
    if any(k in t for k in keywords): return True
    if USE_LLM_FALLBACK:
        resp = ask_gpt(f"Does this sentence mention a person's job or profession? Answer yes or no: '{text}'")
        return 'yes' in resp.lower()
    return False


def detect_sport_in_text(text: str) -> bool:
    t = text.lower()
    keywords = ['sport','play','love','like','running','jogging','swimming','cycling','gym','fitness','football','soccer']
    if any(k in t for k in keywords): return True
    if USE_LLM_FALLBACK:
        resp = ask_gpt(f"Does this sentence describe a sport or physical activity the person does? '{text}' Answer yes or no.")
        return 'yes' in resp.lower()
    return False


def detect_missing_info(text: str, user_name: str) -> List[str]:
    missing = []
    if not detect_weight_in_text(text): missing.append('weight')
    if not detect_job_in_text(text): missing.append('job')
    if not detect_sport_in_text(text): missing.append('sport')
    return missing


def has_complete_profile_info(text: str) -> bool:
    return detect_weight_in_text(text) and detect_job_in_text(text) and detect_sport_in_text(text)


def is_just_greeting_with_name(text: str) -> bool:
    t = text.lower().strip()
    greets = ["hi i'm","hello i'm","hi my name is","hello my name is"]
    for g in greets:
        if g in t and len(text.split()) <= 6 and '?' not in text:
            return True
    w = text.split()
    return len(w)==2 and w[0].lower() in ['hi','hello']

# ------------ Database Operations ------------

def get_user_profile_info(user_name: str) -> Optional[dict]:
    if not mongo_status or not user_name: return None
    try:
        return user_collection.find_one({"name":{"$regex":f"^{user_name}$","$options":"i"}})
    except:
        return None


def load_user_history(name: str) -> List[dict]:
    if not mongo_status or not name: return []
    rec = user_collection.find_one({"name":{"$regex":f"^{name}$","$options":"i"}})
    return rec.get('chat_history',[]) if rec else []


def save_conversation_to_db(user_name: str, human_message: str, ai_message: str) -> bool:
    if not mongo_status or not user_name: return False
    try:
        rec = user_collection.find_one({"name":{"$regex":f"^{user_name}$","$options":"i"}})
        new_msgs = [{"type":"human","content":human_message},{"type":"ai","content":ai_message}]
        if rec:
            hist = rec.get('chat_history',[])
            hist.extend(new_msgs)
            user_collection.update_one({"_id":rec['_id']},{"$set":{"timestamp":datetime.utcnow(),"chat_history":hist}})
            return True
        return False
    except:
        return False


def populate_memory_with_history(memory, chat_history: List[dict]):
    for msg in chat_history:
        if msg['type']=='human': memory.chat_memory.add_user_message(msg['content'])
        else: memory.chat_memory.add_ai_message(msg['content'])

# ------------ Analysis Functions ------------

def analyze_previous_discussions(chat_history: List[dict], user_name: str) -> List[str]:
    topics=[]
    for msg in chat_history:
        if msg['type']=='human':
            c=msg['content'].lower()
            if 'lose weight' in c: topics.append('weight loss')
            if 'gain weight' in c: topics.append('weight gain')
            if 'breakfast' in c: topics.append('breakfast recommendations')
            if 'lunch' in c: topics.append('lunch recommendations')
            if 'diet' in c or 'nutrition' in c: topics.append('diet planning')
            if 'exercise' in c or 'sport' in c: topics.append('exercise advice')
    return list(set(topics))


def check_for_previous_discussion_query(text: str) -> bool:
    keys=['what did we discuss','earlier','previous chat','remind me']
    return any(k in text.lower() for k in keys)

# ------------ Response Generation ------------

def generate_conversational_missing_info_response(user_name: str, missing_info: List[str], user_question: str) -> str:
    if not missing_info: return ''
    greet = f"Hello {user_name}! 👋"
    res = greet + "\n\n"
    if len(missing_info)==3:
        res += "Il me manque quelques infos pour personnaliser mes conseils:\n"
        res += "• Poids (kg)\n• Métier\n• Sport ou activité physique\n"
    elif len(missing_info)==2:
        res += "Presque prêt! Il me manque: " + ", ".join(missing_info)
    else:
        res += "Il manque: " + missing_info[0]
    return res


def generate_profile_summary(user_record: dict, topics: List[str]) -> str:
    if not user_record: return ''
    s=f"Voici ce que je sais:\n• Poids: {user_record.get('weight')} kg\n• Métier: {user_record.get('job')}\n• Sport: {user_record.get('sport')}\n"
    if topics:
        s+="\nSujets précédents:\n"
        for t in topics: s+=f"• {t}\n"
    return s


def generate_previous_discussion_summary(topics: List[str], user_name: str) -> str:
    if not topics: return f"Pas de sujets enregistrés pour toi, {user_name}."
    s=f"{user_name}, on a parlé de:\n"
    for t in topics: s+=f"• {t}\n"
    return s

# ------------ Document & Vector DB ------------

def load_documents():
    docs=[]
    if not os.path.exists(DATA_PATH): return docs
    for fp in glob.glob(os.path.join(DATA_PATH,'*.md')):
        for enc in ['utf-8','latin-1']:
            try:
                loader=TextLoader(fp,encoding=enc)
                docs.extend(loader.load())
                break
            except UnicodeDecodeError:
                continue
    return docs


def split_text(docs):
    splitter=RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=50)
    return [c for c in splitter.split_documents(docs) if c.page_content.strip()]


def save_to_chroma(chunks):
    if os.path.exists(CHROMA_PATH): shutil.rmtree(CHROMA_PATH)
    emb=OpenAIEmbeddings()
    Chroma.from_documents(chunks,emb,persist_directory=CHROMA_PATH)


def ensure_database():
    if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
        docs=load_documents()
        if not docs: return False
        chunks=split_text(docs)
        if not chunks: return False
        save_to_chroma(chunks)
    return True

# ------------ Chain & Main Response ------------

def load_chain_with_history(user_name: Optional[str]=None):
    if not ensure_database(): return None,None,''
    emb=OpenAIEmbeddings()
    vectordb=Chroma(persist_directory=CHROMA_PATH,embedding_function=emb)
    retr=vectordb.as_retriever(search_kwargs={"k":5})
    mem=ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    user_ctx=''
    if user_name and mongo_status:
        rec=get_user_profile_info(user_name)
        if rec:
            user_ctx=f"Profil: {rec['name']}, {rec['weight']}kg, {rec['job']}, {rec['sport']}"
            hist=load_user_history(user_name)
            populate_memory_with_history(mem,hist)
    chain=ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model=MODEL_NAME,temperature=0.2),
        retriever=retr,
        memory=mem
    )
    return chain,mem,user_ctx


def get_assistant_response(prompt_input: str) -> str:
    user_name=extract_name_from_text(prompt_input)
    # Handle greetings and missing info omitted for brevity; implement as desired.
    qa,mem,ctx=load_chain_with_history(user_name)
    if not qa: return "Désolé, la base de données n'est pas prête."
    if ctx:
        q=f"{ctx}. Question: {prompt_input}."
    else:
        q=prompt_input
    res=qa({"question":q})
    ans=res.get("answer","")
    # Optionally save conversation here.
    return ans

# ---------- End of assistant_service.py ----------
