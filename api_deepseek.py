import os
import re
from typing import List

import fastapi
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from openai import APIError, RateLimitError
from pydantic import BaseModel

from rag import LocalRAG, RetrievedChunk

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


ATECO_GROUPS = [
    {"ranges": [(10, 11)], "coeff": "40%"},
    {"ranges": [(45, 45), (46, 46), (47, 47)], "coeff": "40%"},
    {"ranges": [(47, 47)], "coeff": "40%"},
    {"ranges": [(47, 47)], "coeff": "54%"},
    {"ranges": [(41, 43), (68, 68)], "coeff": "86%"},
    {"ranges": [(46, 46)], "coeff": "62%"},
    {"ranges": [(55, 56)], "coeff": "40%"},
    {"ranges": [(64, 66), (69, 75), (85, 85), (86, 88)], "coeff": "78%"},
    {
        "ranges": [
            (1, 3),
            (5, 9),
            (12, 33),
            (35, 35),
            (36, 39),
            (49, 53),
            (58, 63),
            (77, 82),
            (84, 84),
            (90, 93),
            (94, 96),
            (97, 99),
        ],
        "coeff": "67%",
    },
]


def _extract_ateco_prefix(query: str) -> int | None:
    match = re.search(r"\bateco\s*([0-9]{2})(?:[.\s-]?[0-9]{0,2})?\b", query, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"\bcodice\s*([0-9]{2})(?:[.\s-]?[0-9]{0,2})?\b", query, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _lookup_coefficiente_ateco(prefix: int) -> str | None:
    if prefix == 46:
        return (
            "Dipende dal sottocodice ATECO 46: "
            "46.1 = 62%, mentre 46.2-46.9 = 40%."
        )
    if prefix == 47:
        return (
            "Dipende dal sottocodice ATECO 47: "
            "47.81 = 40%, 47.82-47.89 = 54%, "
            "47.1-47.7 e 47.9 = 40%."
        )

    for group in ATECO_GROUPS:
        for start, end in group["ranges"]:
            if start <= prefix <= end:
                return group["coeff"]
    return None


def _is_ateco_coeff_query(query: str) -> bool:
    q = query.lower()
    return (
        "ateco" in q
        and (
            re.search(r"coeffic", q) is not None
            or re.search(r"reddi+tivit", q) is not None
        )
    )


def _is_limit_query(query: str) -> bool:
    q = query.lower()
    return any(term in q for term in ("quanto posso guadagn", "limite", "soglia", "ricavi", "compensi"))


def _is_tax_query(query: str) -> bool:
    q = query.lower()
    return any(term in q for term in ("quanto vengo tass", "tassat", "imposta", "aliquota", "sostitutiva"))


def _is_forfettario_query(query: str) -> bool:
    return "forfett" in query.lower()


def _intent_expansions(query: str) -> List[str]:
    q = query.lower()
    expansions: List[str] = []

    if "ateco" in q:
        expansions.append("tabella coefficienti redditività ateco allegato 4")

    if any(term in q for term in ("soglia", "ricavi", "compensi", "limite", "uscita")):
        expansions.extend(
            [
                "regime forfettario soglia 85000 ricavi compensi",
                "regime forfettario uscita immediata 100000",
                "circolare 32/e 2023 soglie accesso uscita",
            ]
        )

    if any(term in q for term in ("tass", "imposta", "aliquota", "sostitutiva")):
        expansions.extend(
            [
                "regime forfettario imposta sostitutiva 15% 5%",
                "aliquota 5 per cento nuove attività forfettario",
                "quadro lm imposta sostitutiva forfettario",
            ]
        )

    if any(term in q for term in ("scadenz", "saldo", "acconto", "calendario")):
        expansions.extend(
            [
                "calendario fiscale forfettari 2026 saldo acconto",
                "scadenze imposta sostitutiva regime forfettario 2026",
            ]
        )

    if any(term in q for term in ("inps", "contribut", "artigiani", "commercianti", "gestione separata", "35%")):
        expansions.extend(
            [
                "riduzione contributiva 35% regime forfettario",
                "inps artigiani commercianti forfettario 2026",
                "aliquote gestione separata 2026",
            ]
        )

    if any(term in q for term in ("ostativ", "esclusion", "esclus", "cause")):
        expansions.extend(
            [
                "cause ostative regime forfettario 2026",
                "esclusioni regime forfettario lavoro dipendente partecipazioni",
            ]
        )

    return list(dict.fromkeys(expansions))


def _merge_results(primary: List[RetrievedChunk], extras: List[RetrievedChunk], top_k: int = 8) -> List[RetrievedChunk]:
    by_key = {}
    for item in primary + extras:
        key = (item.source, item.chunk_id)
        current = by_key.get(key)
        if current is None or item.score > current.score:
            by_key[key] = item

    ranked = sorted(by_key.values(), key=lambda r: r.score, reverse=True)
    if not ranked:
        return []

    # Favorisce varietà di fonti nei primi risultati per ridurre omissioni su testi tabellari.
    selected: List[RetrievedChunk] = []
    used_sources = set()
    for item in ranked:
        if item.source not in used_sources:
            selected.append(item)
            used_sources.add(item.source)
        if len(selected) >= min(4, top_k):
            break

    for item in ranked:
        key = (item.source, item.chunk_id)
        if any((x.source, x.chunk_id) == key for x in selected):
            continue
        selected.append(item)
        if len(selected) >= top_k:
            break

    return selected


def _search_with_intent(query: str) -> List[RetrievedChunk]:
    primary = rag.search(query, top_k=8, min_score=0.06)
    extra_results: List[RetrievedChunk] = []
    for expanded_query in _intent_expansions(query):
        extra_results.extend(rag.search(expanded_query, top_k=4, min_score=0.05))
    return _merge_results(primary, extra_results, top_k=8)


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

    if _is_ateco_coeff_query(contenuto):
        prefix = _extract_ateco_prefix(contenuto)
        if prefix is not None:
            coeff = _lookup_coefficiente_ateco(prefix)
            if coeff is not None:
                if prefix == 46 or prefix == 47:
                    message = (
                        f"Per il codice ATECO {prefix}, {coeff} "
                        "Controlla sempre il sottocodice completo per il valore esatto. "
                        "Fonti: 03_Tabella_Coefficienti_Redditivita_ATECO.txt, "
                        "01_Legge_190-2014_Base_Normativa_e_Coefficienti.txt"
                    )
                else:
                    message = (
                        f"Il coefficiente di redditività per ATECO {prefix} è {coeff}. "
                        "Fonti: 03_Tabella_Coefficienti_Redditivita_ATECO.txt, "
                        "01_Legge_190-2014_Base_Normativa_e_Coefficienti.txt"
                    )
                return ChatResponse(
                    message=message,
                    sources=[
                        "03_Tabella_Coefficienti_Redditivita_ATECO.txt",
                        "01_Legge_190-2014_Base_Normativa_e_Coefficienti.txt",
                    ],
                )

    if _is_forfettario_query(contenuto) and _is_limit_query(contenuto) and _is_tax_query(contenuto):
        return ChatResponse(
            message=(
                "Per restare nel regime forfettario, nel periodo precedente ricavi o compensi non devono superare 85.000 euro; "
                "se durante l'anno superi 100.000 euro, l'uscita dal regime è immediata. "
                "L'imposta sostitutiva è in via ordinaria al 15%, ridotta al 5% per i primi 5 anni se sono rispettati i requisiti di nuova attività. "
                "Fonti: 02_Circolare_32E-2023_Novita_Soglie_e_Uscita_Immediat.txt, 01_Legge_190-2014_Base_Normativa_e_Coefficienti.txt"
            ),
            sources=[
                "02_Circolare_32E-2023_Novita_Soglie_e_Uscita_Immediat.txt",
                "01_Legge_190-2014_Base_Normativa_e_Coefficienti.txt",
            ],
        )

    retrieved = _search_with_intent(contenuto)
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
