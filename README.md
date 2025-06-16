# 🥗 AI Nutritional Assistant

This is an AI-powered nutritional assistant built with [Streamlit], [LangChain]
, [OpenAI], and [MongoDB]. It gives personalized nutrition advice based on:

- Your **weight**, **job**, and **physical activity**
- **Stored chat history**
- Your **previous goals** (e.g. weight loss, gain)
- Embedded **.md files** for nutrition knowledge (via vector search with Chroma)

---

## 🚀 Features

- Chat interface (Streamlit) with memory
- Personalized responses using OpenAI LLM
- Persistent vector database (Chroma)
- MongoDB integration to store:
  - User profiles
  - Chat history
- Auto-detection of missing info (weight, job, sport)
- Support for multiple users
- Embeddings created once and reused

---

## 📦 Installation

### 1. Clone the repo

```bash
git clone https://github.com/your-username/nutritional-assistant.git
cd nutritional-assistant
