# -*- coding: utf-8 -*-
import os
import shutil
import glob
import re
from datetime import datetime
from urllib.parse import quote_plus
from typing import Literal, List, Optional

import streamlit as st
st.set_page_config(page_title="Nutritional Assistant", page_icon="🥗")

from dotenv import load_dotenv
from pymongo import MongoClient
from pydantic import BaseModel

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()

# Constants
CHROMA_PATH = "chroma"
DATA_PATH = "data/books"
MODEL_NAME = "gpt-3.5-turbo"

# MongoDB Setup
def initialize_mongodb():
    """Initialize MongoDB connection with error handling"""
    try:
        username = "jouininade123"
        password = quote_plus("kookie123")
        uri = f"mongodb+srv://{username}:{password}@cluster0.xr9k99n.mongodb.net/?retryWrites=true&w=majority"
        
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        db = client["nutritional_assistant"]
        user_collection = db["user_info_records"]
        client.server_info()  # Test connection
        
        return client, user_collection, True
    except Exception as e:
        st.error(f"❌ MongoDB connection failed: {e}")
        return None, None, False

# Initialize MongoDB
client, user_collection, mongo_status = initialize_mongodb()

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class UserInfo(BaseModel):
    name: str
    weight: float
    job: str
    sport: str

class ChatMessage(BaseModel):
    type: Literal["human", "ai"]
    content: str

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_name_from_text(text: str) -> Optional[str]:
    """Extract person's name from text using LLM"""
    try:
        name_prompt = PromptTemplate.from_template(
            "Extract only the person's name from this text. Return just the name, nothing else:\n\n{text}"
        )
        model = ChatOpenAI(model=MODEL_NAME, temperature=0)
        name_chain = name_prompt | model
        result = name_chain.invoke({"text": text})
        return result.content.strip()
    except Exception:
        return None

def detect_weight_in_text(text: str) -> bool:
    """Detect if text contains weight information"""
    text_lower = text.lower()
    
    # Weight patterns
    weight_patterns = [
        r'\d+\s*kg',           # "70kg" or "70 kg"
        r'\d+\s*kilo',         # "70kilo" or "70 kilos"
        r'weigh\s+\d+',        # "I weigh 70"
        r'weight\s+is\s+\d+',  # "my weight is 70"
        r'i\s+am\s+\d+\s*kg',  # "I am 70kg"
        r'im\s+\d+\s*kg',      # "I'm 70kg"
    ]
    
    for pattern in weight_patterns:
        if re.search(pattern, text_lower):
            return True
    
    # Check for standalone numbers in weight context
    numbers = re.findall(r'\b(\d+)\b', text_lower)
    for num in numbers:
        if 30 <= int(num) <= 200:  # Reasonable weight range
            num_index = text_lower.find(num)
            surrounding_text = text_lower[max(0, num_index-20):num_index+20]
            if any(weight_word in surrounding_text for weight_word in ['kg', 'kilo', 'weight', 'weigh']):
                return True
    
    return False
def detect_job_in_text(text: str) -> bool:
    """Detect if job/profession info is present; fallback to GPT if needed"""
    text_lower = text.lower()
    job_keywords = [
        'work', 'job', 'profession', 'teacher', 'doctor', 'engineer', 'student', 
        'nurse', 'manager', 'cook', 'chef', 'dancer', 'artist', 'lawyer', 
        'developer', 'designer', 'accountant', 'salesperson', 'mechanic', 
        'pilot', 'writer', 'photographer', 'musician', 'athlete', 'trainer',
        'therapist', 'consultant', 'analyst', 'coordinator', 'assistant',
        'supervisor', 'director', 'executive', 'administrator', 'technician',
        'specialist', 'officer', 'operator', 'clerk', 'representative',
        'advisor', 'counselor', 'instructor', 'professor', 'researcher',
        'scientist', 'pharmacist', 'dentist', 'veterinarian', 'architect',
        'contractor', 'electrician', 'plumber', 'carpenter', 'farmer',
        'waiter', 'bartender', 'cashier', 'receptionist', 'secretary',
        'banker', 'insurance', 'real estate', 'marketing', 'sales', 'finance',
        'hr', 'it', 'security', 'cleaning', 'maintenance', 'delivery',
        'driver', 'transportation', 'logistics', 'warehouse', 'retail',
        'customer service', 'call center', 'freelancer', 'entrepreneur',
        'business owner', 'retired', 'unemployed', 'housewife', 'househusband',
        'stay at home'
    ]

    if any(keyword in text_lower for keyword in job_keywords):
        return True

    # Optional: fallback to LLM if keywords fail
    # Only use if high accuracy is critical
    if USE_LLM_FALLBACK:
        from openai import OpenAI  # or use LangChain/GPT wrapper
        gpt_prompt = f"Does this sentence mention a person's job or profession? Answer yes or no: '{text}'"
        response = ask_gpt(gpt_prompt)  # You'll define this safely
        return "yes" in response.lower()
    
    return False
def detect_sport_in_text(text: str, use_llm_fallback: bool = False) -> bool:
    text_lower = text.lower()
    sport_keywords = [
        'sport', 'play', 'love', 'like', 'enjoy', 'practice', 'do', 'tennis',
        'football', 'soccer', 'basketball', 'volleyball', 'voleyball', 'baseball',
        'running', 'jogging', 'walking', 'hiking', 'swimming', 'cycling', 'biking',
        'dancing', 'boxing', 'martial arts', 'karate', 'judo', 'taekwondo',
        'yoga', 'pilates', 'gym', 'fitness', 'weightlifting', 'bodybuilding',
        'crossfit', 'aerobics', 'zumba', 'spinning', 'climbing', 'skiing',
        'snowboarding', 'skating', 'surfing', 'golf', 'badminton', 'squash',
        'ping pong', 'table tennis', 'chess', 'bowling', 'fishing', 'hunting',
        'horseback riding', 'sailing', 'rowing', 'kayaking', 'canoeing',
        'scuba diving', 'snorkeling', 'parkour', 'skateboarding', 'rollerblading',
        'none', 'nothing', 'no sport', 'dont play', "don't play", 'inactive',
        'sedentary'
    ]

    if any(keyword in text_lower for keyword in sport_keywords):
        return True

    if use_llm_fallback:
        gpt_prompt = (
            f"Does this sentence describe a sport or physical activity the person does regularly or occasionally?\n\n"
            f"Text: \"{text}\"\n\n"
            f"Answer with 'yes' or 'no'."
        )
        response = ask_gpt(gpt_prompt)  # You would define this function
        return 'yes' in response.lower()

    return False

def detect_missing_info(text: str, user_name: str) -> List[str]:
    """Detect what specific information is missing from user input"""
    missing_info = []
    
    if not detect_weight_in_text(text):
        missing_info.append("weight")
    
    if not detect_job_in_text(text):
        missing_info.append("job")
    
    if not detect_sport_in_text(text):
        missing_info.append("sport")
    
    return missing_info

def has_complete_profile_info(text: str) -> bool:
    """Check if the text contains complete profile information"""
    return (detect_weight_in_text(text) and 
            detect_job_in_text(text) and 
            detect_sport_in_text(text))

def is_just_greeting_with_name(user_input: str) -> bool:
    """Check if user input is just a greeting with name"""
    user_input_lower = user_input.lower().strip()
    
    greeting_patterns = [
        "hi i'm", "hello i'm", "hi my name is", "hello my name is",
        "hi i am", "hello i am"
    ]
    
    # Check for greeting introduction patterns
    for pattern in greeting_patterns:
        if pattern in user_input_lower and len(user_input.split()) <= 6 and "?" not in user_input:
            return True
    
    # Check for simple "hi [name]" or "hello [name]"
    words = user_input.split()
    if len(words) == 2 and words[0].lower() in ["hi", "hello"]:
        return True
        
    return False

# ============================================================================
# DATABASE OPERATIONS
# ============================================================================

def get_user_profile_info(user_name: str) -> Optional[dict]:
    """Get user profile information from database"""
    if not mongo_status or not user_name:
        return None
        
    try:
        user_record = user_collection.find_one({
            "name": {"$regex": f"^{user_name}$", "$options": "i"}
        })
        return user_record
    except Exception as e:
        st.error(f"Error loading user profile: {e}")
        return None

def load_user_history(name: str) -> List[dict]:
    """Load user's chat history from database"""
    if not mongo_status or not name:
        return []
    
    try:
        user_record = user_collection.find_one({
            "name": {"$regex": f"^{name}$", "$options": "i"}
        })
        if user_record and "chat_history" in user_record:
            return user_record["chat_history"]
    except Exception as e:
        st.error(f"Error loading user history: {e}")
    
    return []

def save_conversation_to_db(user_name: str, human_message: str, ai_message: str) -> bool:
    """Save a single conversation exchange to the database"""
    if not mongo_status or not user_name:
        return False
    
    try:
        user_record = user_collection.find_one({
            "name": {"$regex": f"^{user_name}$", "$options": "i"}
        })
        
        new_messages = [
            {"type": "human", "content": human_message},
            {"type": "ai", "content": ai_message}
        ]
        
        if user_record:
            # Update existing user's chat history
            existing_chat = user_record.get("chat_history", [])
            existing_chat.extend(new_messages)
            
            user_collection.update_one(
                {"_id": user_record["_id"]},
                {"$set": {
                    "timestamp": datetime.utcnow(),
                    "chat_history": existing_chat
                }}
            )
            
            return True
        else:
            st.warning("⚠️ User profile not found. Please provide complete profile information first.")
            return False
            
    except Exception as e:
        st.error(f"Error saving conversation: {e}")
        return False

def populate_memory_with_history(memory, chat_history: List[dict]):
    """Populate conversation memory with chat history"""
    for msg in chat_history:
        if msg["type"] == "human":
            memory.chat_memory.add_user_message(msg["content"])
        elif msg["type"] == "ai":
            memory.chat_memory.add_ai_message(msg["content"])

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_previous_discussions(chat_history: List[dict], user_name: str) -> List[str]:
    """Analyze chat history to extract key topics discussed"""
    topics_discussed = []
    
    for msg in chat_history:
        if msg["type"] == "human":
            content = msg["content"].lower()
            
            # Weight loss discussions
            if any(phrase in content for phrase in ["lose weight", "loose weight", "weight loss", "want to lose"]):
                topics_discussed.append("weight loss")
            
            # Weight gain discussions
            if any(phrase in content for phrase in ["gain weight", "want to gain", "increase weight", "put on weight"]):
                topics_discussed.append("weight gain")
            
            # Meal discussions
            if any(phrase in content for phrase in ["breakfast", "breakfset", "morning meal", "petit déjeuner"]):
                topics_discussed.append("breakfast recommendations")
            
            if any(phrase in content for phrase in ["lunch", "déjeuner", "midday meal"]):
                topics_discussed.append("lunch recommendations")
            
            # Diet discussions
            if any(phrase in content for phrase in ["diet", "nutrition", "meal plan", "eating"]):
                topics_discussed.append("diet planning")
            
            # Exercise discussions
            if any(phrase in content for phrase in ["exercise", "workout", "training", "sport", "running", "dancing"]):
                topics_discussed.append("exercise advice")
    
    return list(set(topics_discussed))  # Remove duplicates

def check_for_previous_discussion_query(user_input: str) -> bool:
    """Check if user is asking about previous conversations"""
    previous_discussion_keywords = [
        "what did we discuss", "what we talked about", "our previous conversation",
        "what did i ask", "earlier", "before", "last time", "remind me",
        "what we discussed", "previous chat"
    ]
    
    user_input_lower = user_input.lower()
    return any(keyword in user_input_lower for keyword in previous_discussion_keywords)

# ============================================================================
# RESPONSE GENERATION
# ============================================================================

def generate_conversational_missing_info_response(user_name: str, missing_info: List[str], user_question: str) -> str:
    """Generate a conversational response asking for missing information"""
    if not missing_info:
        return ""
    
    # Greeting based on user's question tone
    if any(greeting in user_question.lower() for greeting in ['hi', 'hello', 'hey']):
        greeting = f"Hi {user_name}! Nice to meet you! 😊"
    else:
        greeting = f"Hello {user_name}! 👋"
    
    response = greeting + "\n\n"
    
    if len(missing_info) == 3:  # Missing weight, job, and sport
        response += "I'd love to help you with personalized nutrition advice! To give you the best recommendations, I need to know a bit more about you.\n\n"
        response += "Could you tell me:\n"
        response += "• What's your current weight? (in kg)\n"
        response += "• What do you do for work?\n"
        response += "• Do you play any sports or have favorite physical activities?\n\n"
        response += "For example: *'I'm 65kg, I work as a teacher, and I love jogging in the mornings.'*"
        
    elif len(missing_info) == 2:
        response += "Great! I have some information about you already. "
        
        if "weight" in missing_info and "job" in missing_info:
            response += "Could you also tell me your weight (in kg) and what you do for work? This helps me give you better nutrition advice! 😊"
        elif "weight" in missing_info and "sport" in missing_info:
            response += "Could you also tell me your weight (in kg) and what sports or activities you enjoy? This helps me personalize my advice! 🏃‍♀️"
        elif "job" in missing_info and "sport" in missing_info:
            response += "Could you also tell me what you do for work and what sports or activities you enjoy? This helps me understand your lifestyle better! 💼"
            
    elif len(missing_info) == 1:
        response += "Almost there! Just one more thing - "
        
        if "weight" in missing_info:
            response += "could you tell me your current weight in kg? This helps me give you accurate portion and calorie recommendations! ⚖️"
        elif "job" in missing_info:
            response += "what do you do for work? This helps me understand your daily activity level and schedule! 💼"
        elif "sport" in missing_info:
            response += "do you play any sports or have favorite physical activities? Even if it's just walking, I'd love to know! 🚶‍♀️"
    
    response += "\n\nOnce I have this info, I can give you much better personalized nutrition advice! 🥗"
    
    return response

def generate_profile_summary(user_record: dict, topics_discussed: List[str]) -> str:
    """Generate a personalized profile summary"""
    if not user_record:
        return ""
        
    name = user_record.get("name", "")
    weight = user_record.get("weight", "")
    job = user_record.get("job", "")
    sport = user_record.get("sport", "")
    
    summary = f"Here's what I know about you:\n\n"
    summary += f"• **Weight**: {weight} kg\n"
    summary += f"• **Job**: {job}\n" 
    summary += f"• **Sport**: {sport}\n\n"
    
    if topics_discussed:
        summary += "**Previous discussions:**\n"
        for topic in topics_discussed:
            if topic == "weight gain":
                summary += "• You asked about **weight gain** strategies\n"
            elif topic == "weight loss":
                summary += "• You asked about **weight loss** recommendations\n" 
            elif topic == "breakfast recommendations":
                summary += "• You asked about **breakfast** recommendations\n"
            elif topic == "lunch recommendations":
                summary += "• You asked about **lunch** recommendations\n"
            elif topic == "diet planning":
                summary += "• We discussed **diet planning** and nutrition\n"
            elif topic == "exercise advice":
                summary += f"• We talked about **exercise** advice for your {sport}\n"
    else:
        summary += "This is our first detailed conversation! Feel free to ask me about nutrition, diet plans, or meal recommendations. 😊"
    
    return summary

def generate_previous_discussion_summary(topics_discussed: List[str], user_name: str) -> str:
    """Generate a summary of previous discussions"""
    if not topics_discussed:
        return f"I don't have any record of specific nutrition topics from our previous discussions, {user_name}."
    
    summary = f"Based on our previous conversations, {user_name}, here's what we discussed:\n\n"
    
    for topic in topics_discussed:
        if topic == "weight gain":
            summary += "• You asked about **weight gain** - you mentioned wanting to gain weight\n"
        elif topic == "weight loss":
            summary += "• You asked about **weight loss** - you mentioned wanting to lose weight\n"
        elif topic == "breakfast recommendations":
            summary += "• You asked about **breakfast** recommendations\n"
        elif topic == "lunch recommendations":
            summary += "• You asked about **lunch** recommendations\n"
        elif topic == "diet planning":
            summary += "• We discussed **diet planning** and nutrition\n"
        elif topic == "exercise advice":
            summary += "• We talked about **exercise** and workout advice related to your activities\n"
    
    return summary

# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

def load_documents():
    """Load documents from the data directory"""
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
    """Split documents into chunks"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    return [chunk for chunk in chunks if chunk.page_content.strip() and len(chunk.page_content) > 20]

def save_to_chroma(chunks):
    """Save document chunks to Chroma vector database"""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    embedding = OpenAIEmbeddings()
    Chroma.from_documents(chunks, embedding, persist_directory=CHROMA_PATH)

def ensure_database():
    """Ensure the vector database exists and is populated"""
    if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
        docs = load_documents()
        if not docs:
            return False
        
        chunks = split_text(docs)
        if not chunks:
            return False
        
        save_to_chroma(chunks)
    
    return True

# ============================================================================
# CHAIN SETUP
# ============================================================================

def load_chain_with_history(user_name: Optional[str] = None):
    """Load the conversational retrieval chain with user history"""
    if not ensure_database():
        return None, None, ""

    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Load user profile for context
    user_context = ""
    if user_name:
        user_record = get_user_profile_info(user_name)
        if user_record:
            user_context = f"User Profile - Name: {user_record.get('name', '')}, Weight: {user_record.get('weight', '')} kg, Job: {user_record.get('job', '')}, Sport: {user_record.get('sport', '')}"
            st.success(f"✅ Found user profile: {user_record.get('name', '')} - {user_record.get('weight', '')}kg")
        else:
            st.warning("⚠️ No user profile found in database")
            
        chat_history = load_user_history(user_name)
        if chat_history:
            populate_memory_with_history(memory, chat_history)
            st.info(f"💭 Welcome back, {user_name}! I remember our previous conversations.")

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model=MODEL_NAME, temperature=0.2),
        retriever=retriever,
        memory=memory,
        verbose=False
    )
    
    return qa_chain, memory, user_context

# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_user_input(prompt_input: str):
    """Main function to process user input and handle different scenarios"""
    
    user_name = extract_name_from_text(prompt_input.strip())
    user_question = prompt_input.strip()
    user_record = get_user_profile_info(user_name) if user_name else None

    # 1. Handle greeting only (no question)
    if is_just_greeting_with_name(user_question) and user_name:
        if user_record:
            # Existing user greeting
            chat_history = user_record.get("chat_history", [])
            topics_discussed = analyze_previous_discussions(chat_history, user_name)
            profile_summary = generate_profile_summary(user_record, topics_discussed)
            greeting_response = f"Hello {user_name}! Great to see you again! 😊\n\n{profile_summary}\n\nWhat nutrition question can I help you with today?"

            st.markdown("### 👤 Welcome Back!")
            st.markdown(f"<div style='background:#f0f8ff;color:#000;padding:1.5rem;border-radius:10px;border-left:4px solid #2196F3;margin:1rem 0;'>{greeting_response}</div>", unsafe_allow_html=True)
            save_conversation_to_db(user_name, user_question, greeting_response)
        else:
            # New user - ask for missing info conversationally
            missing_info = ["weight", "job", "sport"]
            conversational_response = generate_conversational_missing_info_response(user_name, missing_info, user_question)
            st.markdown("### 💬 Assistant's Response:")
            st.markdown(f"<div style='background:#f0f8ff;color:#000;padding:1.5rem;border-radius:10px;border-left:4px solid #4CAF50;margin:1rem 0;'>{conversational_response}</div>", unsafe_allow_html=True)
        return

    # 2. Handle missing information for new users
    if user_name and not user_record:
        missing_info = detect_missing_info(user_question, user_name)
        if missing_info:
            conversational_response = generate_conversational_missing_info_response(user_name, missing_info, user_question)
            st.markdown("### 💬 Assistant's Response:")
            st.markdown(f"<div style='background:#f0f8ff;color:#000;padding:1.5rem;border-radius:10px;border-left:4px solid #4CAF50;margin:1rem 0;'>{conversational_response}</div>", unsafe_allow_html=True)
            save_conversation_to_db(user_name, user_question, conversational_response)
            return

    # 3. Check for prior discussion summary request
    if check_for_previous_discussion_query(user_question) and user_name:
        chat_history = load_user_history(user_name)
        topics_discussed = analyze_previous_discussions(chat_history, user_name)
        summary = generate_previous_discussion_summary(topics_discussed, user_name)
        st.markdown("### 🕘 Previous Discussion Summary:")
        st.markdown(f"<div style='background:#e8f4fd;color:#000;padding:1.5rem;border-radius:10px;border-left:4px solid #4CAF50;margin:1rem 0;'>{summary}</div>", unsafe_allow_html=True)
        save_conversation_to_db(user_name, user_question, summary)
        return

    # 4. Load QA chain and process nutrition question
    qa_chain, memory, user_context = load_chain_with_history(user_name)
    if not qa_chain:
        st.error("❌ No markdown files found in data/books.")
        return

    # 5. Display user's question
    with st.spinner("🤔 The assistant is analyzing your question..."):
        st.markdown("### 🙋‍♀️ Your Question:")
        st.markdown(f"<div style='background:#f8f9fa;color:#000;padding:1rem;border-radius:8px;border-left:3px solid #007bff;margin:1rem 0;'><strong>You asked:</strong> {user_question}</div>", unsafe_allow_html=True)

        # Generate response
        if user_context and user_name and user_record:
            # Handle simple user profile queries directly
            lower_q = user_question.lower()
            if "weight" in lower_q and "what" in lower_q:
                advice_text = f"Your weight is {user_record.get('weight')} kg, {user_name}."
            elif "job" in lower_q and "what" in lower_q:
                advice_text = f"Your job is {user_record.get('job')}, {user_name}."
            elif "sport" in lower_q and "what" in lower_q:
                advice_text = f"Your sport is {user_record.get('sport')}, {user_name}."
            elif "name" in lower_q and "what" in lower_q:
                advice_text = f"Your name is {user_name}."
            else:
                # Enhanced question with user context
                enhanced_question = f"User profile: {user_name}, {user_record.get('weight')} kg, works as {user_record.get('job')}, plays {user_record.get('sport')}. Question: {user_question}. Provide personalized nutrition advice."
                result = qa_chain({"question": enhanced_question})
                advice_text = result["answer"]
        else:
            # Fallback to generic QA
            result = qa_chain({"question": user_question})
            advice_text = result["answer"]

    # 6. Display response
    st.markdown("### 💬 Expert's Response:")
    st.markdown(f"<div style='background:#f8f9fa;color:#000;padding:1.5rem;border-radius:10px;border-left:4px solid #667eea;margin:1rem 0;'>{advice_text}</div>", unsafe_allow_html=True)

    # 7. Save conversation
    if user_name:
        save_conversation_to_db(user_name, user_question, advice_text)

    # 8. Try extracting structured user info if complete
    if not is_just_greeting_with_name(user_question) and has_complete_profile_info(user_question):
        try:
            parser = PydanticOutputParser(pydantic_object=UserInfo)
            prompt_template = PromptTemplate.from_template(
                "Extract the user information from the following text. If any information is missing, skip this extraction:\n\n{text}\n\n{format_instructions}",
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            model = ChatOpenAI(model=MODEL_NAME, temperature=0)
            structured_chain = prompt_template | model | parser

            user_info = structured_chain.invoke({"text": prompt_input.strip()})
            st.markdown("### 📋 Structured User Info:")
            st.json(user_info.model_dump())

            if mongo_status:
                chat_messages = [
                    ChatMessage(type=msg.type, content=msg.content).model_dump()
                    for msg in memory.chat_memory.messages
                ]
                existing_user = user_collection.find_one({
                    "name": {"$regex": f"^{user_info.name}$", "$options": "i"}
                })

                if existing_user:
                    user_collection.update_one(
                        
                        {"_id": existing_user["_id"]},
                        {"$set": {
                            "weight": user_info.weight,
                            "job": user_info.job,
                            "sport": user_info.sport,
                            "timestamp": datetime.utcnow(),
                            "chat_history": chat_messages
                        }}
                    )
                    st.success("✅ Updated your profile and conversation history!")
                else:
                    user_collection.insert_one({
                        "name": user_info.name,
                        "weight": user_info.weight,
                        "job": user_info.job,
                        "sport": user_info.sport,
                        "timestamp": datetime.utcnow(),
                        "chat_history": chat_messages
                    })
                    st.success("✅ Created your profile and saved conversation history!")
        except Exception as e:
            st.error(f"⚠️ Couldn't extract structured data: {e}")


# ---- Footer ----
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666; padding: 1rem;'>🥗 Nutritional Assistant - Powered by AI & LangChain</div>", unsafe_allow_html=True)
# Add this code at the END of your existing file, after all your functions

# ============================================================================
# STREAMLIT UI INTERFACE
# ============================================================================

# Add this code at the END of your existing file, after all your functions

# ============================================================================
# STREAMLIT UI INTERFACE
# ============================================================================
def get_assistant_response(prompt_input: str) -> str:
    """Modified version of process_user_input that returns response instead of displaying it"""
    
    user_name = extract_name_from_text(prompt_input.strip())
    user_question = prompt_input.strip()
    user_record = get_user_profile_info(user_name) if user_name else None
    
    # Update current user in session state
    if user_name:
        st.session_state.current_user = user_name

    # 1. Handle greeting only (no question)
    if is_just_greeting_with_name(user_question) and user_name:
        if user_record:
            # Existing user greeting
            chat_history = user_record.get("chat_history", [])
            topics_discussed = analyze_previous_discussions(chat_history, user_name)
            profile_summary = generate_profile_summary(user_record, topics_discussed)
            greeting_response = f"Hello {user_name}! Great to see you again! 😊\n\n{profile_summary}\n\nWhat nutrition question can I help you with today?"
            save_conversation_to_db(user_name, user_question, greeting_response)
            return greeting_response
        else:
            # New user - ask for missing info conversationally
            missing_info = ["weight", "job", "sport"]
            conversational_response = generate_conversational_missing_info_response(user_name, missing_info, user_question)
            return conversational_response

    # 2. Handle missing information for new users
    if user_name and not user_record:
        missing_info = detect_missing_info(user_question, user_name)
        if missing_info:
            conversational_response = generate_conversational_missing_info_response(user_name, missing_info, user_question)
            save_conversation_to_db(user_name, user_question, conversational_response)
            return conversational_response

    # 3. Check for prior discussion summary request
    if check_for_previous_discussion_query(user_question) and user_name:
        chat_history = load_user_history(user_name)
        topics_discussed = analyze_previous_discussions(chat_history, user_name)
        summary = generate_previous_discussion_summary(topics_discussed, user_name)
        save_conversation_to_db(user_name, user_question, summary)
        return summary

    # 4. Load QA chain and process nutrition question
    qa_chain, memory, user_context = load_chain_with_history(user_name)
    if not qa_chain:
        return "❌ Sorry, I couldn't access the nutrition knowledge base. Please make sure the data files are available."

    # Generate response
    try:
        if user_context and user_name and user_record:
            # Handle simple user profile queries directly
            lower_q = user_question.lower()
            if "weight" in lower_q and "what" in lower_q:
                advice_text = f"Your weight is {user_record.get('weight')} kg, {user_name}."
            elif "job" in lower_q and "what" in lower_q:
                advice_text = f"Your job is {user_record.get('job')}, {user_name}."
            elif "sport" in lower_q and "what" in lower_q:
                advice_text = f"Your sport is {user_record.get('sport')}, {user_name}."
            elif "name" in lower_q and "what" in lower_q:
                advice_text = f"Your name is {user_name}."
            else:
                # Enhanced question with user context
                enhanced_question = f"User profile: {user_name}, {user_record.get('weight')} kg, works as {user_record.get('job')}, plays {user_record.get('sport')}. Question: {user_question}. Provide personalized nutrition advice."
                result = qa_chain({"question": enhanced_question})
                advice_text = result["answer"]
        else:
            # Fallback to generic QA
            result = qa_chain({"question": user_question})
            advice_text = result["answer"]
        
        # Save conversation to database
        if user_name:
            save_conversation_to_db(user_name, user_question, advice_text)
        
        # Try extracting structured user info if complete
        if not is_just_greeting_with_name(user_question) and has_complete_profile_info(user_question):
            try:
                parser = PydanticOutputParser(pydantic_object=UserInfo)
                prompt_template = PromptTemplate.from_template(
                    "Extract the user information from the following text. If any information is missing, skip this extraction:\n\n{text}\n\n{format_instructions}",
                    partial_variables={"format_instructions": parser.get_format_instructions()}
                )
                model = ChatOpenAI(model=MODEL_NAME, temperature=0)
                structured_chain = prompt_template | model | parser

                user_info = structured_chain.invoke({"text": prompt_input.strip()})
                
                if mongo_status:
                    chat_messages = [
                        ChatMessage(type=msg.type, content=msg.content).model_dump()
                        for msg in memory.chat_memory.messages
                    ]
                    existing_user = user_collection.find_one({
                        "name": {"$regex": f"^{user_info.name}$", "$options": "i"}
                    })

                    if existing_user:
                        user_collection.update_one(
                            {"_id": existing_user["_id"]},
                            {"$set": {
                                "weight": user_info.weight,
                                "job": user_info.job,
                                "sport": user_info.sport,
                                "timestamp": datetime.utcnow(),
                                "chat_history": chat_messages
                            }}
                        )
                        advice_text += "\n\n✅ Updated your profile and conversation history!"
                    else:
                        user_collection.insert_one({
                            "name": user_info.name,
                            "weight": user_info.weight,
                            "job": user_info.job,
                            "sport": user_info.sport,
                            "timestamp": datetime.utcnow(),
                            "chat_history": chat_messages
                        })
                        advice_text += "\n\n✅ Created your profile and saved conversation history!"
            except Exception as e:
                advice_text += f"\n\n⚠️ Note: Couldn't extract structured data: {e}"
        
        return advice_text
        
    except Exception as e:
        return f"❌ Sorry, I encountered an error: {str(e)}"
def main():
    """Main Streamlit application interface"""
    
    # Header
    st.title("🥗 Nutritional Assistant")
    st.markdown("*Your AI-powered nutrition companion*")
    
    # Sidebar with app info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This AI nutritional assistant provides personalized advice based on:
        - Your weight and fitness goals
        - Your job and lifestyle  
        - Your sports and activities
        - Evidence-based nutrition knowledge
        """)
        
        # Display MongoDB connection status
        if mongo_status:
            st.success("✅ Database Connected")
        else:
            st.error("❌ Database Disconnected")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize session state for current user
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about nutrition, diet plans, or tell me about yourself..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get the response using your existing logic
                response = get_assistant_response(prompt)
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
# Add this line at the very end of your file (after the main() function definition)

if __name__ == "__main__":
    main()