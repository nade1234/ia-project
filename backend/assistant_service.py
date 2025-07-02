import os
import re
import json
from datetime import datetime
from typing import List, Optional
from urllib.parse import quote_plus

from dotenv import load_dotenv
from pymongo import MongoClient
from pydantic import BaseModel

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma

# --- Toggle for LLM fallbacks ---
USE_LLM_FALLBACK = True

# --- Tokens we never want as ingredients ---
BAD_INGREDIENTS = {
    "want","to","gain","weight","i","am","eat","lose","have",
    "and","me","my","for","a","the","of","with","is","it","on","at",
    "anymore","any","more"
}

def clean_ingredients(ings: List[str]) -> List[str]:
    """Strip, lowercase, filter out BAD_INGREDIENTS, then dedupe preserving order."""
    cleaned = []
    for it in ings:
        w = it.strip().lower()
        if len(w)>1 and w not in BAD_INGREDIENTS and w.isalpha():
            if w not in cleaned:
                cleaned.append(w)
    return cleaned

def detect_removed_ingredients(text: str) -> List[str]:
    """
    Look for "don't have X, Y anymore" or "no X, Y" patterns
    and return a list of those items (cleaned & deduped).
    """
    text_l = text.lower()
    removed = []

    # Pattern: don't have X, Y anymore
    m = re.search(r"dont have\s+([\w\s,]+?)(?:\s+any ?more|$)", text_l)
    if m:
        parts = re.split(r",|\s+and\s+", m.group(1))
        removed += parts

    # Pattern: no X, Y
    m2 = re.search(r"\bno\s+([\w\s,]+?)(?:\s+any ?more|$)", text_l)
    if m2:
        parts = re.split(r",|\s+and\s+", m2.group(1))
        removed += parts

    return clean_ingredients(removed)

# --- Environment & MongoDB setup ---
load_dotenv()
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma")
MODEL_NAME  = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

def initialize_mongodb():
    try:
        user = os.getenv("MONGO_USER")
        pwd  = quote_plus(os.getenv("MONGO_PASS") or "")
        host = os.getenv("MONGO_HOST")
        uri  = f"mongodb+srv://{user}:{pwd}@{host}/?retryWrites=true&w=majority"
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        db     = client[os.getenv("MONGO_DB","nutritional_assistant")]
        coll   = db[os.getenv("MONGO_COLLECTION","user_info_records")]
        client.server_info()
        return client, coll, True
    except Exception as e:
        print("MongoDB init error:", e)
        return None, None, False

client, user_collection, mongo_status = initialize_mongodb()

# --- Data Model ---
class UserInfo(BaseModel):
    ingredients: List[str]    = []
    fitness_goal: Optional[str]    = None
    number_of_meals: Optional[int] = None

# --- LLM‐based extraction helpers ---
def detect_ingredients_in_text(text: str) -> List[str]:
    prompt = PromptTemplate.from_template(
        "Extract only the food ingredients as a comma-separated list "
        "(no sentences, just ingredients). Example: 'I have tomatoes and chicken' -> tomatoes, chicken. Text: {text}"
    )
    resp = ChatOpenAI(model=MODEL_NAME, temperature=0) \
           .invoke(prompt.format(text=text)).content
    items = [i for i in resp.split(",") if i.strip()]
    return clean_ingredients(items)

def detect_fitness_goal_in_text(text: str) -> Optional[str]:
    kws = ['lose weight','weight loss','muscle gain','toning','maintain','endurance']
    low = text.lower()
    for kw in kws:
        if kw in low:
            return kw
    if USE_LLM_FALLBACK:
        prompt = PromptTemplate.from_template(
            "Does this text express a fitness goal? If yes, return just the goal, else 'none': {text}"
        )
        out = ChatOpenAI(model=MODEL_NAME, temperature=0) \
              .invoke(prompt.format(text=text)).content.strip()
        return out if out.lower()!='none' else None
    return None

def detect_number_of_meals_in_text(text: str) -> Optional[int]:
    m = re.search(r"(\d+)\s*(meals?|plates?)", text.lower())
    if m: return int(m.group(1))
    if USE_LLM_FALLBACK:
        prompt = PromptTemplate.from_template(
            "Extract the number of meals per day or 'none': {text}"
        )
        out = ChatOpenAI(model=MODEL_NAME, temperature=0) \
              .invoke(prompt.format(text=text)).content.strip()
        return int(out) if out.isdigit() else None
    return None

def detect_missing_info(ui: UserInfo) -> List[str]:
    miss = []
    if not ui.ingredients:     miss.append('ingredients')
    if not ui.fitness_goal:    miss.append('fitness goal')
    if not ui.number_of_meals: miss.append('number of meals')
    return miss

def generate_missing_info_prompt(missing: List[str], lang: str="en") -> str:
    if lang=="en":
        lines = "\n".join(f"• {m}" for m in missing)
        return f"I need the following information:\n{lines}\nCould you provide them?"
    else:
        lines = "\n".join(f"• {m}" for m in missing)
        return f"Il me manque ces informations :\n{lines}\nPeux-tu me les indiquer ?"

# --- Profile CRUD ---
def get_user_profile(name: str) -> Optional[dict]:
    if not mongo_status or not name: return None
    return user_collection.find_one({"name":{"$regex":f"^{name}$","$options":"i"}})

def save_user_profile(name: str, info: UserInfo) -> bool:
    if not mongo_status: return False
    doc = {"name":name, **info.dict(), "timestamp":datetime.utcnow()}
    try:
        user_collection.update_one(
            {"name":{"$regex":f"^{name}$","$options":"i"}},
            {"$set":doc}, upsert=True
        )
        return True
    except Exception as e:
        print("DB update error:", e)
        return False

# --- RAG + Memory chain loader ---
def load_chain_with_history(user_name: Optional[str]=None):
    emb      = OpenAIEmbeddings(model=MODEL_NAME)
    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=emb)
    retr     = vectordb.as_retriever(search_kwargs={"k":5})
    mem      = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain    = ConversationalRetrievalChain.from_llm(
                   llm=ChatOpenAI(model=MODEL_NAME, temperature=0.2),
                   retriever=retr, memory=mem
               )
    return chain, mem, ""

def agent2_generate_mealplan(ui: UserInfo, days:int=7) -> str:
    meals      = ui.number_of_meals or 2
    label_sets = {
        1:["Meal"], 2:["Lunch","Dinner"],
        3:["Breakfast","Lunch","Dinner"],
        4:["Breakfast","Lunch","Snack","Dinner"]
    }
    labels     = label_sets.get(meals,[f"Meal {i+1}" for i in range(meals)])
    meal_lines = "\n".join(f"- {lbl} :" for lbl in labels)

    prompt = (
        f"I have ONLY these ingredients: {', '.join(ui.ingredients)}. "
        f"My goal: {ui.fitness_goal or 'N/A'}. I want {meals} meals/day. "
        f"Plan for {days} days:\nDay 1:\n{meal_lines}\n(Continue …)\nUse ONLY these ingredients."
    )
    try:
        qa,_,_ = load_chain_with_history(None)
        return qa({"question":prompt}).get("answer","").strip()
    except:
        return ChatOpenAI(model=MODEL_NAME, temperature=0.5).invoke(prompt).content.strip()

# --- Master agent ---
def master_agent(user_name:str, user_input:str, days:int=7) -> str:
    rec = get_user_profile(user_name) or {}
    old = UserInfo(
        ingredients    = rec.get("ingredients",[]),
        fitness_goal   = rec.get("fitness_goal"),
        number_of_meals= rec.get("number_of_meals")
    )

    # 1) Extract any new info
    new = extract_userinfo_from_text(user_input)

    # 2) Combine old + additions
    combined = old.ingredients + new.ingredients

    # 3) Detect removals & subtract
    removed = detect_removed_ingredients(user_input)
    filtered= [i for i in combined if i not in removed]

    # 4) Clean & dedupe
    final_ings = clean_ingredients(filtered)

    # 5) Build merged profile
    merged = UserInfo(
        ingredients     = final_ings,
        fitness_goal    = new.fitness_goal or old.fitness_goal,
        number_of_meals = new.number_of_meals or old.number_of_meals
    )

    # 6) Save if anything changed
    if new.ingredients or new.fitness_goal or new.number_of_meals or removed:
        save_user_profile(user_name, merged)

    # 7) Ask for missing info
    miss = detect_missing_info(merged)
    if miss:
        return generate_missing_info_prompt(miss)

    # 8) If it's a meal-plan ask, generate
    if any(kw in user_input.lower() for kw in ["plan","menu","what to eat","give me"]):
        return agent2_generate_mealplan(merged, days)

    return "Your profile has been updated."

def get_assistant_response(prompt_input:str, user_name:str="user", days:int=7)->str:
    return master_agent(user_name, prompt_input, days)

def extract_userinfo_from_text(text:str)->UserInfo:
    prompt = PromptTemplate.from_template("""
You are an expert assistant. Extract ONLY these fields as JSON:
ingredients (list), fitness_goal (string), number_of_meals (integer).
Don’t include non-food words. If only goal or only number, return only that.

Input: {text}

Examples (reply exactly like these):
{{"ingredients": ["chicken","eggs"]}}
or
{{"fitness_goal": "gain weight"}}
or
{{"number_of_meals": 2}}
""")
    resp = ChatOpenAI(model=MODEL_NAME, temperature=0) \
           .invoke(prompt.format(text=text)).content
    try:
        data = json.loads(resp)
        return UserInfo(
            ingredients     = clean_ingredients(data.get("ingredients",[])),
            fitness_goal    = data.get("fitness_goal"),
            number_of_meals = data.get("number_of_meals")
        )
    except:
        return UserInfo()
