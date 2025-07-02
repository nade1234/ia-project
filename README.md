🥗 AI Nutritional Assistant

An AI-powered nutritional assistant built with Streamlit
, LangChain
, OpenAI
, and MongoDB
.

It provides personalized nutrition advice based on:

Your weight, job, and physical activity

Your previous goals (e.g., weight loss, gain)

Stored chat history for contextualized conversations

Embedded knowledge base (Markdown files, stored in a Chroma vector DB)

🚀 Features

🗨️ Chat interface (Streamlit) with conversational memory

🤖 Personalized responses with OpenAI LLM

📚 Knowledge embeddings stored in Chroma (created once and reused)

🗄️ MongoDB integration to store:

User profiles

Chat history

🔍 Auto-detection of missing info (weight, job, sport)

👥 Multi-user support

📸 Screenshots
🔹 Registration & Profile Setup
<img src="5c6c2545-2d68-4002-9fbe-b4640aa50127.jpg" width="600">
🔹 AI Chat Interface
<img src="8a99dba6-58c6-459e-a192-5191d251c80b.jpg" width="600">
🔹 Nutrition Goals & History
<img src="ed5f8907-7517-4ad7-b4ed-9e809447e251.jpg" width="600">
📦 Installation
1. Clone the repo
git clone https://github.com/your-username/nutritional-assistant.git
cd nutritional-assistant

2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

3. Install dependencies
pip install -r requirements.txt

4. Set up environment variables

Create a .env file with:

OPENAI_API_KEY=your_openai_api_key
MONGO_URI=your_mongodb_connection_string

5. Run the app
streamlit run front.py