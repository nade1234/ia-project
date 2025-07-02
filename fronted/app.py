import streamlit as st
import requests

# URL of your FastAPI backend
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Nutritional Assistant", page_icon="🥗")
st.title("🥗 Nutritional Assistant")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input area
if prompt := st.chat_input("Ask me about nutrition, diet plans, or tell me about yourself..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        # Call the FastAPI backend
        try:
            response = requests.post(
                f"{API_URL}/chat", json={"prompt": prompt}, timeout=15
            )
            response.raise_for_status()
            answer = response.json().get("response", "")
        except Exception as e:
            answer = f"❌ Error contacting the server: {e}"
        
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
