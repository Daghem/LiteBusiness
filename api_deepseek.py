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
chiave_api = os.getenv("API_KEY_DEEPSEEK")
if not chiave_api:
    raise ValueError("Manca API_KEY_DEEPSEEK nel file .env")

llm_model = "deepseek-chat"
client = OpenAI(
    api_key=chiave_api,
    base_url="https://api.deepseek.com",
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
        "Se il CONTEXT non contiene una parte della risposta, dillo in una sola frase breve. "
        "Non inventare norme, soglie o scadenze. "
        "Stile obbligatorio: italiano chiaro, tono professionale, nessun markdown, "
        "nessun uso di **, # o elenchi con trattini. "
        "Non iniziare con formule tipo 'In base al CONTEXT fornito'."
    )

    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"DOMANDA:\n{contenuto}\n\n"
                        f"CONTEXT:\n{context}\n\n"
                        "Rispondi in italiano in modo sintetico. "
                        "Usa massimo 4 frasi totali. "
                        "Dai prima la risposta diretta. "
                        "Chiudi con 'Fonti: ...' usando solo i nomi file, separati da virgola."
                    ),
                },
            ],
            stream=False,
        )
    except RateLimitError:
        return ChatResponse(
            message=(
                "Quota DeepSeek esaurita o limite raggiunto (errore 429). "
                "Controlla piano e billing del provider selezionato."
            ),
            sources=[],
        )
    except APIError as error:
        return ChatResponse(
            message=f"Errore API DeepSeek: {error}",
            sources=[],
        )

    answer = response.choices[0].message.content or ""
    sources = list(dict.fromkeys(item.source for item in retrieved))
    return ChatResponse(message=answer, sources=sources)
