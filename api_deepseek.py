from httpx import request
from openai import OpenAI
import os
from dotenv import load_dotenv
import fastapi
from fastapi.middleware.cors import CORSMiddleware

load_dotenv() # Carica le variabili dal file .env
chiave_api = os.getenv("API_KEY_DEEPSEEK")


client = OpenAI(
    api_key=chiave_api, 
    base_url="https://api.deepseek.com"
)




app = fastapi.FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In produzione metterai l'URL del tuo sito
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/")
async def read_root(request: fastapi.Request):
    data = await request.json()
    contenuto = data.get("content")
    response = client.chat.completions.create(
    model="deepseek-chat", # Usa "deepseek-chat" per risposte più veloci e meno costose
    messages=[
        {"role": "system", "content": "Sei un esperto di diritto tributario italiano."},
        {"role": "user", "content": contenuto}
    ],
    stream=False
    )
    
    return {"message": response.choices[0].message.content}
    

