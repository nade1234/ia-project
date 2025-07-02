FROM python:3.11-slim

# ─── Prevent any proxy settings from leaking in ───
ENV HTTP_PROXY="" \
    HTTPS_PROXY="" \
    NO_PROXY="*"





WORKDIR /app

# copy & install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy backend code
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
