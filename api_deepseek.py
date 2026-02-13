import os
from typing import List

import fastapi
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from openai import APIError, RateLimitError
from pydantic import BaseModel

from rag import LocalRAG

load_dotenv()  # Carica le variabili dal file .env
chiave_api = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=chiave_api,
)

app = fastapi.FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In produzione imposta l'URL del tuo frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = LocalRAG()
rag_load_error = None
try:
    rag.load()
except Exception as error:  # pragma: no cover
    rag_load_error = str(error)


class ChatRequest(BaseModel):
    content: str


class ChatResponse(BaseModel):
    message: str
    sources: List[str]


@app.post("/", response_model=ChatResponse)
async def read_root(payload: ChatRequest):
    if rag_load_error:
        return ChatResponse(
            message=(
                "Indice RAG non disponibile. Esegui prima: "
                "`python3 build_rag_index.py`."
            ),
            sources=[],
        )

    contenuto = payload.content.strip()
    if not contenuto:
        return ChatResponse(message="Inserisci una domanda valida.", sources=[])

    retrieved = rag.search(contenuto, top_k=5, min_score=0.08)
    if not retrieved:
        return ChatResponse(
            message=(
                "Non trovo informazioni pertinenti nei documenti forniti. "
                "Riformula la domanda o aggiungi documentazione."
            ),
            sources=[],
        )

    context_blocks = []
    for item in retrieved:
        context_blocks.append(
            (
                f"[Fonte: {item.source} | Chunk: {item.chunk_id} | "
                f"Score: {item.score:.3f}]\n{item.text}"
            )
        )

    context = "\n\n".join(context_blocks)
    system_prompt = (
        "Sei un assistente fiscale per il regime forfettario italiano. "
        "Rispondi solo con informazioni presenti nel CONTEXT. "
        "Se il CONTEXT non contiene la risposta, dillo esplicitamente. "
        "Non inventare norme, soglie o scadenze."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"DOMANDA:\n{contenuto}\n\n"
                        f"CONTEXT:\n{context}\n\n"
                        "Rispondi in italiano in modo sintetico e cita le fonti usate."
                    ),
                },
            ],
            stream=False,
        )
    except RateLimitError:
        return ChatResponse(
            message=(
                "Quota OpenAI esaurita o limite raggiunto (errore 429). "
                "Controlla billing/plan su platform.openai.com oppure usa una chiave con credito."
            ),
            sources=[],
        )
    except APIError as error:
        return ChatResponse(
            message=f"Errore API OpenAI: {error}",
            sources=[],
        )

    answer = response.choices[0].message.content or ""
    sources = list(dict.fromkeys(item.source for item in retrieved))
    return ChatResponse(message=answer, sources=sources)
