# backend/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import traceback                # ← new

from assistant_service import get_assistant_response

app = FastAPI(title="Nutritional Assistant API")

class ChatRequest(BaseModel):
    prompt: str
    user_name: Optional[str] = "user"

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        answer = get_assistant_response(
            prompt_input=request.prompt,
            user_name=request.user_name
        )
        return ChatResponse(response=answer)
    except Exception as e:
        traceback.print_exc()     # ← this will dump the full stack to your uvicorn console
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}
