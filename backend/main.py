from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Import or implement the core logic for generating responses
# Here we assume you have refactored get_assistant_response and related functions
# into a module named `assistant_service`.
from assistant_service import get_assistant_response

app = FastAPI(title="Nutritional Assistant API")

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint to process a nutrition chat prompt and return a response.
    """
    try:
        answer = get_assistant_response(request.prompt)
        return ChatResponse(response=answer)
    except Exception as e:
        # Return a 500 error if anything goes wrong
        raise HTTPException(status_code=500, detail=str(e))

# Optionally, add a healthcheck endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}
