import streamlit as st
import requests

# (Defined here so you won't see NameError if you reference it.)
USE_LLM_FALLBACK = True

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Nutritional Assistant", page_icon="🥗")
st.title("🥗 Nutritional Assistant")

if "username" not in st.session_state:
    st.session_state.username = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

st.sidebar.text_input("Your name", key="username")
username = st.session_state.username.strip().lower()

if not username:
    st.warning("Please enter your name in the sidebar.")
else:
    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle new prompt
    if prompt := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                resp = requests.post(
                    f"{API_URL}/chat",
                    json={"prompt": prompt, "user_name": username},
                    timeout=15
                )
                resp.raise_for_status()
                answer = resp.json().get("response", "")
            except Exception as e:
                answer = f"❌ Error contacting the server: {e}"

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
