import json
import os
import re
import secrets
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import List

import fastapi
from app_paths import DATA_ROOT, DOCUMENT_ROOTS, FRONTEND_ROOT, LOG_DIR, RAG_INDEX_PATH, UPLOADS_ROOT
from app_models import (
    ChatRequest,
    ChatResponse,
    ChatSummary,
    ChatTranscript,
    ChatTurnPayload,
    FeedbackRequest,
    FeedbackResponse,
    RegimeOption,
    SimulationRequest,
    SimulationResponse,
    SourceRef,
)
from dotenv import load_dotenv
from fastapi import File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from openai import OpenAI
from openai import APIError, RateLimitError
from storage_services import ChatHistoryStore, EventStore, FeedbackStore, build_admin_stats
from tax_simulator import simulate_forfettario

from lexical_fallback import LexicalChunk, LexicalFallbackIndex
from rag_qdrant import CorpusConfig, QdrantRAG, RetrievedChunk

load_dotenv()  # Carica le variabili dal file .env
chiave_api = os.getenv("API_KEY_DEEPSEEK", "").strip()

llm_model = "deepseek-chat"
client: OpenAI | None = None
client_init_error: str | None = None

def _get_llm_client() -> OpenAI | None:
    global client, client_init_error
    if client is not None:
        return client
    if client_init_error is not None:
        return None
    if not chiave_api:
        client_init_error = "API_KEY_DEEPSEEK non configurata sul server."
        return None
    try:
        client = OpenAI(
            api_key=chiave_api,
            base_url="https://api.deepseek.com",
        )
    except Exception as error:  # pragma: no cover
        client_init_error = f"Impossibile inizializzare DeepSeek: {error}"
        return None
    return client

app = fastapi.FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In produzione imposta l'URL del tuo frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = QdrantRAG.from_env()
rag_load_error = None
rag_ready = False

LEXICAL_FALLBACK_ENABLED = os.getenv("LEXICAL_FALLBACK_ENABLED", "1") != "0"
SEMANTIC_SEARCH_ENABLED = os.getenv("SEMANTIC_SEARCH_ENABLED", "1") != "0"
LOG_RAG_EVENTS = os.getenv("LOG_RAG_EVENTS", "0") == "1"
chat_store = ChatHistoryStore(DATA_ROOT / "chat_history")
feedback_store = FeedbackStore(DATA_ROOT / "feedback" / "feedback.jsonl")
event_store = EventStore(DATA_ROOT / "events" / "app_events.jsonl")
ADMIN_ACCESS_KEY = os.getenv("ADMIN_ACCESS_KEY", "").strip()
FRONTEND_PAGES = {"index.html", "chat.html", "dashboard.html", "admin.html", "admin_tools.html"}
FRONTEND_ASSETS = {"admin.css", "style.css", "style_home.css", "logo.png", "robot.png"}
FORFETTARIO_REGIME_ID = "forfettario"
FORFETTARIO_LABEL = "Regime Forfettario"
FORFETTARIO_ALIASES = (
    "forfettario",
    "forfettari",
    "regime forfettario",
    "regime dei forfettari",
)
FORFETTARIO_CORPUS_DIRNAME = "Normativo_Forfettari_Agg_2026"


def _discover_corpora() -> List[CorpusConfig]:
    for root in DOCUMENT_ROOTS:
        candidate = root / FORFETTARIO_CORPUS_DIRNAME
        if not candidate.is_dir():
            continue
        if any(candidate.rglob("*.pdf")) or any(candidate.rglob("*.xml")):
            return [QdrantRAG.derive_corpus_config(candidate)]
    return []

def _log_rag_event(event: str, payload: dict) -> None:
    event_store.append({"event": event, **payload})
    if not LOG_RAG_EVENTS:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    log_path = LOG_DIR / "rag_events.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

def _require_admin(admin_key: str | None) -> None:
    if not ADMIN_ACCESS_KEY:
        raise HTTPException(
            status_code=503,
            detail="ADMIN_ACCESS_KEY non configurata sul server.",
        )
    if not admin_key or not secrets.compare_digest(admin_key, ADMIN_ACCESS_KEY):
        raise HTTPException(status_code=401, detail="Credenziali admin non valide.")

def _build_lexical_index(regime_ids: List[str]) -> LexicalFallbackIndex | None:
    if rag_load_error or not rag_ready:
        return None
    chunks: List[LexicalChunk] = []
    try:
        for payload in rag.iter_payload_chunks(regime_ids=regime_ids, batch_size=256):
            regime = payload.get("regime")
            source = payload.get("source")
            chunk_id = payload.get("chunk_id")
            text = payload.get("text")
            if regime is None or source is None or chunk_id is None or text is None:
                continue
            chunks.append(
                LexicalChunk(
                    regime=str(regime),
                    source=str(source),
                    chunk_id=int(chunk_id),
                    text=str(text),
                    page_start=int(payload.get("page_start"))
                    if payload.get("page_start") is not None
                    else None,
                    page_end=int(payload.get("page_end"))
                    if payload.get("page_end") is not None
                    else None,
                )
            )
    except Exception:
        return None
    if not chunks:
        return None
    return LexicalFallbackIndex.from_chunks(chunks)
@dataclass(frozen=True)
class RegimeProfile:
    regime_id: str
    label: str
    aliases: tuple[str, ...]
    is_default: bool = False

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

def _build_regime_profiles() -> List[RegimeProfile]:
    return [
        RegimeProfile(
            regime_id=FORFETTARIO_REGIME_ID,
            label=FORFETTARIO_LABEL,
            aliases=FORFETTARIO_ALIASES,
            is_default=True,
        )
    ]

def _build_regime_aliases(corpus: CorpusConfig) -> tuple[str, ...]:
    tokens = [token for token in corpus.regime_id.split("_") if token]
    aliases = {corpus.regime_id.replace("_", " "), corpus.label.lower()}
    aliases.update(tokens)
    if "regime" not in tokens:
        aliases.add(f"regime {corpus.regime_id.replace('_', ' ')}")
    if corpus.regime_id == "forfettario":
        aliases.update(
            (
                "forfettario",
                "forfettari",
                "regime forfettario",
                "regime dei forfettari",
            )
        )
    return tuple(sorted({alias.strip() for alias in aliases if alias.strip()}, key=len, reverse=True))

REGIME_PROFILES: List[RegimeProfile] = []
DEFAULT_REGIME_ID = FORFETTARIO_REGIME_ID

def _refresh_regime_profiles() -> None:
    global REGIME_PROFILES, DEFAULT_REGIME_ID
    REGIME_PROFILES = _build_regime_profiles()
    DEFAULT_REGIME_ID = next(
        (profile.regime_id for profile in REGIME_PROFILES if profile.is_default),
        REGIME_PROFILES[0].regime_id,
    )

_refresh_regime_profiles()

lexical_index: LexicalFallbackIndex | None = None
if LEXICAL_FALLBACK_ENABLED:
    lexical_index = LexicalFallbackIndex.from_local_index(RAG_INDEX_PATH)

def _ensure_rag_ready() -> bool:
    global rag_ready, rag_load_error
    if rag_ready:
        return True
    try:
        rag.load()
    except Exception as error:  # pragma: no cover
        rag_load_error = str(error)
        return False
    rag_load_error = None
    rag_ready = True
    return True

def _compact_excerpt(text: str, max_length: int = 220) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 1].rstrip() + "…"

def _build_source_details(items: List[RetrievedChunk]) -> List[SourceRef]:
    details: List[SourceRef] = []
    seen = set()
    for item in items:
        key = (item.source, item.chunk_id)
        if key in seen:
            continue
        details.append(
            SourceRef(
                source=item.source,
                excerpt=_compact_excerpt(item.text),
                chunk_id=item.chunk_id,
                page_start=getattr(item, "page_start", None),
                page_end=getattr(item, "page_end", None),
                score=round(item.score, 4),
            )
        )
        seen.add(key)
    return details[:4]

def _confidence_from_results(
    retrieved: List[RetrievedChunk],
    retrieval_mode: str,
) -> tuple[str | None, float | None]:
    if not retrieved:
        return None, None
    top_score = max(item.score for item in retrieved)
    if retrieval_mode == "lexical":
        if top_score >= 0.18:
            return "media", round(top_score, 4)
        return "bassa", round(top_score, 4)
    if top_score >= 0.3:
        return "alta", round(top_score, 4)
    if top_score >= 0.18:
        return "media", round(top_score, 4)
    return "bassa", round(top_score, 4)

def _respond(
    message: str,
    sources: List[str] | None = None,
    *,
    source_details: List[SourceRef] | None = None,
    confidence_label: str | None = None,
    confidence_score: float | None = None,
    retrieval_mode: str | None = None,
    regime_id: str | None = None,
    chat_id: str | None = None,
) -> ChatResponse:
    return ChatResponse(
        message=message,
        sources=sources or [],
        source_details=source_details or [],
        confidence_label=confidence_label,
        confidence_score=confidence_score,
        retrieval_mode=retrieval_mode,
        regime_id=regime_id,
        chat_id=chat_id,
    )

def _normalize_match_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.replace("’", "'")
    return normalized

def _tokenize_for_matching(text: str) -> list[str]:
    return re.findall(r"[a-z0-9%]+", _normalize_match_text(text))

def _is_close_alias_token(token: str, alias_token: str) -> bool:
    if token == alias_token:
        return True
    if len(token) < 5 or len(alias_token) < 5:
        return False
    if any(char.isdigit() for char in token + alias_token):
        return False
    if abs(len(token) - len(alias_token)) > 2:
        return False
    if token[0] != alias_token[0] or token[-1] != alias_token[-1]:
        return False
    return SequenceMatcher(None, token, alias_token).ratio() >= 0.82

def _query_matches_alias(query: str, alias: str) -> bool:
    normalized_query = _normalize_match_text(query)
    normalized_alias = _normalize_match_text(alias)
    if normalized_alias in normalized_query:
        return True

    alias_tokens = [token for token in normalized_alias.split() if token]
    query_tokens = _tokenize_for_matching(normalized_query)
    if not alias_tokens or not query_tokens:
        return False
    if len(alias_tokens) == 1:
        return any(_is_close_alias_token(token, alias_tokens[0]) for token in query_tokens)

    window_size = len(alias_tokens)
    for start in range(len(query_tokens) - window_size + 1):
        window = query_tokens[start : start + window_size]
        if all(
            candidate == expected or _is_close_alias_token(candidate, expected)
            for candidate, expected in zip(window, alias_tokens)
        ):
            return True
    return False

def _query_mentions_regime_id(query: str, regime_id: str) -> bool:
    profile = next((item for item in REGIME_PROFILES if item.regime_id == regime_id), None)
    if profile is None:
        return False
    return any(_query_matches_alias(query, alias) for alias in profile.aliases)

EXACT_QUERY_TOKEN_REPLACEMENTS = {
    "forchettario": "forfettario",
    "forfetario": "forfettario",
    "forfetarrio": "forfettario",
    "forfettarrio": "forfettario",
    "forfetaio": "forfettario",
    "alliquota": "aliquota",
    "aliquuota": "aliquota",
    "aligquota": "aliquota",
    "sogllia": "soglia",
    "sogglia": "soglia",
    "intrastad": "intrastat",
    "intrastatto": "intrastat",
    "intrastatt": "intrastat",
    "vieds": "vies",
    "veis": "vies",
    "viess": "vies",
    "bolllo": "bollo",
    "inpss": "inps",
    "atteco": "ateco",
    "atceo": "ateco",
    "contibuti": "contributi",
    "contributtiva": "contributiva",
    "contributtivi": "contributivi",
    "fatturrato": "fatturato",
    "fattturato": "fatturato",
    "incasato": "incassato",
    "incasssato": "incassato",
    "scadneza": "scadenza",
    "domnada": "domanda",
    "impsota": "imposta",
    "extraue": "extra ue",
    "partitaiva": "partita iva",
}

CANONICAL_QUERY_TOKENS = (
    "forfettario",
    "aliquota",
    "soglia",
    "intrastat",
    "contributi",
    "contributiva",
    "contributivi",
    "fatturato",
    "incassato",
    "scadenza",
    "domanda",
    "imposta",
    "sostitutiva",
    "agevolazione",
    "artigiani",
    "commercianti",
    "partecipazioni",
    "controllo",
    "societa",
    "requisiti",
    "ricavi",
    "compensi",
    "dicitura",
    "riduzione",
    "ateco",
    "forfettari",
    "naspi",
    "residenza",
    "cassa",
    "integrativo",
    "detrazioni",
)

def _canonicalize_tax_token(match: re.Match[str]) -> str:
    token = match.group(0)
    exact_replacement = EXACT_QUERY_TOKEN_REPLACEMENTS.get(token)
    if exact_replacement is not None:
        return exact_replacement
    if len(token) < 6 or any(char.isdigit() for char in token):
        return token

    best_match = token
    best_score = 0.0
    for candidate in CANONICAL_QUERY_TOKENS:
        if token[0] != candidate[0] or token[-1] != candidate[-1]:
            continue
        if abs(len(token) - len(candidate)) > 2:
            continue
        score = SequenceMatcher(None, token, candidate).ratio()
        if score >= 0.84 and score > best_score:
            best_match = candidate
            best_score = score
    return best_match

def _normalize_tax_query(query: str) -> str:
    normalized = _normalize_match_text(query)
    normalized = re.sub(r"[a-z0-9%]+", _canonicalize_tax_token, normalized)
    return re.sub(r"\s+", " ", normalized).strip()

DEFINITION_PATTERNS = (
    r"(?:cos'?e|cosa e|che cos'?e|che cosa e)",
    r"(?:definizione di|definisci)",
    r"(?:cosa significa|che significa|significa)",
)

def _extract_definition_term(query: str) -> str | None:
    q = _normalize_tax_query(query)
    if not any(re.search(pattern, q) for pattern in DEFINITION_PATTERNS):
        return None

    patterns = [
        r"(?:cos'?e|cosa e|che cos'?e|che cosa e)\s+(?:il|lo|la|l')?\s*([^?.,;]+)",
        r"(?:definizione di|definisci)\s+(?:il|lo|la|l')?\s*([^?.,;]+)",
        r"(?:cosa significa|che significa|significa)\s+(?:il|lo|la|l')?\s*([^?.,;]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            term = match.group(1).strip()
            term = re.sub(r"\s+", " ", term)
            term = term.strip(" \"'")
            if 1 <= len(term.split()) <= 6:
                return term
    return None

def _term_appears_in_text(term: str, text: str) -> bool:
    if not term or not text:
        return False
    pattern = re.compile(rf"\\b{re.escape(term)}\\b", flags=re.IGNORECASE)
    return pattern.search(text) is not None

def _collect_term_mentions(term: str, regime_id: str) -> List[LexicalChunk]:
    if lexical_index is None:
        return []
    return lexical_index.find_mentions(term, regime_id=regime_id)

def _definition_fallback_message(term: str) -> str:
    return (
        f"Il termine {term} è citato nei documenti disponibili ma non viene definito. "
        "Se vuoi una definizione, aggiungi una fonte che lo spieghi oppure chiedi "
        "una risposta generale senza vincoli di fonte."
    )
def _extract_ateco_components(query: str) -> tuple[int, int | None] | None:
    match = re.search(
        r"\bateco\s*([0-9]{2})(?:[.\s-]?([0-9]{1,2}))?(?=[^0-9]|$)",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(
            r"\bcodice\s*([0-9]{2})(?:[.\s-]?([0-9]{1,2}))?(?=[^0-9]|$)",
            query,
            flags=re.IGNORECASE,
        )
    if not match:
        return None
    prefix = int(match.group(1))
    subcode = int(match.group(2)) if match.group(2) else None
    return prefix, subcode

def _lookup_coefficiente_ateco(prefix: int, subcode: int | None = None) -> str | None:
    if prefix == 46:
        if subcode is not None:
            if subcode == 1 or 10 <= subcode <= 19:
                return "62%"
            return "40%"
        return (
            "Dipende dal sottocodice ATECO 46: "
            "46.1 = 62%, mentre 46.2-46.9 = 40%."
        )
    if prefix == 47:
        if subcode is not None:
            if subcode == 81:
                return "40%"
            if 82 <= subcode <= 89:
                return "54%"
            if 10 <= subcode <= 79 or 90 <= subcode <= 99:
                return "40%"
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

def _classify_query_relevance(query: str, regime_label: str) -> str:
    """Usa il LLM per classificare la query.

    Restituisce:
        'pertinente'    — la domanda riguarda temi fiscali/contributivi
        'off_topic'     — la domanda non c'entra con il regime fiscale
        'ateco_lookup'  — la domanda chiede il coefficiente di redditività ATECO
    """
    llm_client = _get_llm_client()
    if llm_client is None:
        return "pertinente"

    try:
        response = llm_client.chat.completions.create(
            model=llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Sei un classificatore per un assistente fiscale sul {regime_label} italiano. "
                        "Classifica la domanda dell'utente rispondendo con UNA SOLA parola:\n"
                        "ATECO_LOOKUP — se l'utente chiede il coefficiente di redditività, la redditività, "
                        "o la percentuale associata a un codice ATECO specifico (es. 'coefficiente ATECO 69', "
                        "'redditività del codice 47.82', 'quanto è la redditività ATECO 62').\n"
                        "OFF_TOPIC — se la domanda è completamente estranea al mondo fiscale, contabile "
                        "e lavorativo (es. sport, meteo, cucina, intrattenimento, saluti come 'come stai').\n"
                        "PERTINENTE — per QUALSIASI domanda che riguardi, anche indirettamente, "
                        "fisco, tasse, contributi, IVA, INPS, fatturazione, regimi fiscali, partita IVA, "
                        "codici ATECO, coefficienti di redditività, reddito, imposta sostitutiva, "
                        "detrazioni, deduzioni, soglie, ricavi, compensi, aliquote, "
                        "contabilità, dichiarazioni, scadenze fiscali, o qualsiasi altro concetto "
                        "economico-fiscale. Nel dubbio, classifica come PERTINENTE."
                    ),
                },
                {"role": "user", "content": query},
            ],
            stream=False,
            max_tokens=5,
        )
        label = (response.choices[0].message.content or "").strip().upper()
        if label in ("PERTINENTE", "OFF_TOPIC", "ATECO_LOOKUP"):
            return label.lower()
    except Exception:
        pass
    return "pertinente"




def _match_regime_profiles(query: str) -> List[RegimeProfile]:
    matches: List[RegimeProfile] = []
    for profile in REGIME_PROFILES:
        if any(_query_matches_alias(query, alias) for alias in profile.aliases):
            matches.append(profile)
    return matches

def _resolve_regime(query: str) -> tuple[RegimeProfile | None, bool, bool]:
    matches = _match_regime_profiles(query)
    if not matches:
        default_profile = next(
            (profile for profile in REGIME_PROFILES if profile.regime_id == DEFAULT_REGIME_ID),
            REGIME_PROFILES[0] if REGIME_PROFILES else None,
        )
        return default_profile, False, False

    unique_matches = {profile.regime_id: profile for profile in matches}
    return next(iter(unique_matches.values())), True, False

def _regime_scope_message(active_regime: RegimeProfile | None = None) -> str:
    if active_regime is None:
        return "Posso aiutarti solo sul regime forfettario."
    return (
        "Posso aiutarti solo su temi fiscali e contributivi legati alla documentazione caricata"
        f" per {active_regime.label.lower()}."
    )

def _clean_model_answer(answer: str) -> str:
    cleaned = answer.strip()
    replacements = (
        (r"\bcontesto fornito\b", "documenti disponibili"),
        (r"\bCONTEXT\b", "contesto"),
        (r"\bIl contesto fornito non contiene\b", "I documenti disponibili non contengono"),
        (r"\bIl contesto fornito non riporta\b", "I documenti disponibili non riportano"),
        (r"\bIl contesto fornito non specifica\b", "I documenti disponibili non specificano"),
        (r"\bIl contesto fornito non fornisce\b", "I documenti disponibili non forniscono"),
        (r"\bIl contesto non contiene\b", "I documenti disponibili non contengono"),
        (r"\bIl contesto non riporta\b", "I documenti disponibili non riportano"),
        (r"\bIl contesto non specifica\b", "I documenti disponibili non specificano"),
        (r"\bIl contesto non fornisce\b", "I documenti disponibili non forniscono"),
    )
    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned

def _intent_expansions(query: str, regime_id: str) -> List[str]:
    if regime_id != "forfettario":
        return []
    q = _normalize_tax_query(query)
    expansions: List[str] = []

    definition_term = _extract_definition_term(q)
    if definition_term:
        expansions.extend(
            [
                f"definizione {definition_term}",
                f"cos'e {definition_term}",
                f"{definition_term} significato",
            ]
        )

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
                "domanda riduzione contributiva 35 entro 28 febbraio",
                "scadenza domanda agevolazione contributiva artigiani commercianti",
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

def _dynamic_score_thresholds(query: str) -> List[float]:
    token_count = len(_normalize_tax_query(query).split())
    if token_count <= 3:
        return [0.16, 0.12, 0.08]
    if token_count <= 6:
        return [0.2, 0.14, 0.1]
    return [0.22, 0.18, 0.12]

def _search_with_intent(query: str, regime_id: str) -> tuple[List[RetrievedChunk], str]:
    normalized_query = _normalize_tax_query(query)
    primary_queries = [normalized_query]
    if query.strip() and normalized_query != query.strip().lower():
        primary_queries.append(query.strip())

    lexical_results: List[RetrievedChunk] = []
    if lexical_index is not None:
        lexical_hits = lexical_index.search(normalized_query, top_k=6, regime_id=regime_id)
        lexical_results = [
            RetrievedChunk(
                regime=chunk.regime,
                source=chunk.source,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=min(score + 0.04, 1.0),
                page_start=chunk.page_start,
                page_end=chunk.page_end,
            )
            for chunk, score in lexical_hits
        ]

    if not SEMANTIC_SEARCH_ENABLED:
        if lexical_results:
            return lexical_results, "lexical"
        return [], "none"

    thresholds = _dynamic_score_thresholds(normalized_query)
    for threshold in thresholds:
        primary_results: List[RetrievedChunk] = []
        for primary_query in primary_queries:
            primary_results.extend(
                rag.search(
                    primary_query,
                    top_k=8,
                    min_score=threshold,
                    regime_ids=[regime_id],
                )
            )

        extra_results: List[RetrievedChunk] = []
        for expanded_query in _intent_expansions(normalized_query, regime_id=regime_id):
            extra_results.extend(
                rag.search(
                    expanded_query,
                    top_k=4,
                    min_score=max(threshold - 0.02, 0.05),
                    regime_ids=[regime_id],
                )
            )
        merged = _merge_results(primary_results, extra_results + lexical_results, top_k=8)
        if merged:
            mode = "hybrid" if lexical_results else "semantic"
            return merged, mode

    if not lexical_results:
        return [], "none"
    return lexical_results, "lexical"

def _resolve_requested_regime(regime_id: str | None) -> RegimeProfile | None:
    if not regime_id:
        return None
    normalized = QdrantRAG.normalize_regime_id(regime_id)
    return next((item for item in REGIME_PROFILES if item.regime_id == normalized), None)

def _reload_runtime_indexes() -> None:
    global rag, rag_load_error, rag_ready, lexical_index
    _refresh_regime_profiles()
    rag = QdrantRAG.from_env()
    rag.load()
    rag_load_error = None
    rag_ready = True
    if LEXICAL_FALLBACK_ENABLED:
        regime_ids = [profile.regime_id for profile in REGIME_PROFILES]
        lexical_index = _build_lexical_index(regime_ids)
    else:
        lexical_index = None

@app.get("/regimes", response_model=List[RegimeOption])
async def list_regimes():
    _refresh_regime_profiles()
    return [
        RegimeOption(
            regime_id=profile.regime_id,
            label=profile.label,
            is_default=profile.is_default,
        )
        for profile in REGIME_PROFILES
    ]

@app.post("/simulate", response_model=SimulationResponse)
async def simulate(payload: SimulationRequest):
    if payload.regime_id != FORFETTARIO_REGIME_ID:
        raise HTTPException(
            status_code=400,
            detail="Il simulatore disponibile in questa versione copre solo il regime forfettario.",
        )
    try:
        return simulate_forfettario(payload)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

@app.post("/chat-history", response_model=ChatTranscript)
async def persist_chat_turn(payload: ChatTurnPayload):
    chat = chat_store.save_turn(
        chat_id=payload.chat_id,
        regime_id=payload.regime_id,
        user_message=payload.user_message,
        assistant_message=payload.assistant_message,
        assistant_sources=payload.assistant_sources,
        confidence_label=payload.confidence_label,
        confidence_score=payload.confidence_score,
        retrieval_mode=payload.retrieval_mode,
    )
    event_store.append(
        {
            "event": "chat_turn_saved",
            "chat_id": payload.chat_id,
            "regime_id": payload.regime_id,
            "confidence_label": payload.confidence_label,
            "confidence_score": payload.confidence_score,
            "retrieval_mode": payload.retrieval_mode,
        }
    )
    return ChatTranscript(
        chat_id=chat["chat_id"],
        title=chat["title"],
        regime_id=chat.get("regime_id"),
        created_at=chat.get("created_at", ""),
        updated_at=chat.get("updated_at", ""),
        messages=chat["messages"],
    )

@app.get("/chat-history", response_model=List[ChatSummary])
async def list_chat_history():
    return chat_store.list_chats()

@app.get("/chat-history/{chat_id}", response_model=ChatTranscript)
async def get_chat_history(chat_id: str):
    transcript = chat_store.get_chat(chat_id)
    if transcript is None:
        raise HTTPException(status_code=404, detail="Chat non trovata.")
    return transcript

@app.delete("/chat-history/{chat_id}", response_model=FeedbackResponse)
async def delete_chat_history(chat_id: str):
    deleted = chat_store.delete_chat(chat_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat non trovata.")
    return FeedbackResponse(status="deleted")

@app.post("/feedback", response_model=FeedbackResponse)
async def save_feedback(payload: FeedbackRequest):
    vote = payload.vote.strip().lower()
    if vote not in {"up", "down"}:
        raise HTTPException(status_code=400, detail="Vote non valido.")
    feedback_store.append(payload.model_dump())
    return FeedbackResponse(status="saved")

@app.get("/admin/overview")
async def admin_overview(x_admin_key: str | None = Header(default=None)):
    _require_admin(x_admin_key)
    stats = build_admin_stats(chat_store, feedback_store, event_store)
    recent_feedback = feedback_store.read_all()[-10:]
    return {
        "stats": stats.model_dump(),
        "recent_feedback": recent_feedback,
        "recent_chats": [item.model_dump() for item in chat_store.list_chats(limit=10)],
    }

@app.post("/admin/auth/verify")
async def admin_auth_verify(x_admin_key: str | None = Header(default=None)):
    _require_admin(x_admin_key)
    return {"status": "authorized"}

@app.post("/admin/upload")
async def admin_upload_document(
    file: UploadFile = File(...),
    regime_id: str | None = None,
    x_admin_key: str | None = Header(default=None),
):
    _require_admin(x_admin_key)
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".xml"}:
        raise HTTPException(status_code=400, detail="Sono supportati solo PDF e XML.")
    target_regime = QdrantRAG.normalize_regime_id(regime_id or DEFAULT_REGIME_ID)
    if target_regime != FORFETTARIO_REGIME_ID:
        raise HTTPException(
            status_code=400,
            detail="FlyTax supporta solo il regime forfettario.",
        )
    target_dir = UPLOADS_ROOT / FORFETTARIO_CORPUS_DIRNAME
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / Path(filename).name
    target_path.write_bytes(await file.read())
    _refresh_regime_profiles()
    return {"status": "uploaded", "path": str(target_path), "regime_id": target_regime}

@app.post("/admin/reindex")
async def admin_reindex(x_admin_key: str | None = Header(default=None)):
    _require_admin(x_admin_key)
    corpora = _discover_corpora()
    if not corpora:
        raise HTTPException(
            status_code=400,
            detail=f"Nessuna cartella {FORFETTARIO_CORPUS_DIRNAME} con PDF/XML trovata.",
        )
    total_chunks = rag.build_from_pdf_directories(
        corpora=corpora,
        chunk_size=1200,
        overlap=200,
        embed_batch_size=32,
        recreate_collection=True,
    )
    _reload_runtime_indexes()
    event_store.append(
        {
            "event": "reindex_completed",
            "total_chunks": total_chunks,
            "regime_id": FORFETTARIO_REGIME_ID,
        }
    )
    return {
        "status": "reindexed",
        "total_chunks": total_chunks,
        "regime_id": FORFETTARIO_REGIME_ID,
    }

@app.get("/", include_in_schema=False, response_class=FileResponse)
async def serve_home():
    return FileResponse(FRONTEND_ROOT / "index.html")

@app.get("/healthz", include_in_schema=False)
async def healthcheck():
    return {
        "status": "ok",
        "rag_ready": rag_ready,
        "rag_load_error": rag_load_error,
        "semantic_search_enabled": SEMANTIC_SEARCH_ENABLED,
    }

@app.post("/", response_model=ChatResponse)
async def read_root(payload: ChatRequest):
    if SEMANTIC_SEARCH_ENABLED and not _ensure_rag_ready():
        return _respond(
            message=(
                "Indice RAG su Qdrant non disponibile. Verifica `QDRANT_URL` e la "
                "collection configurata, oppure esegui `python3 build_rag_index.py` "
                "prima del deploy."
            ),
            sources=[],
            chat_id=payload.chat_id,
        )

    raw_contenuto = payload.content.strip()
    if not raw_contenuto:
        return _respond(
            message="Inserisci una domanda valida.",
            sources=[],
            chat_id=payload.chat_id,
        )
    contenuto = _normalize_tax_query(raw_contenuto)

    requested_regime = _resolve_requested_regime(payload.regime_id)
    if payload.regime_id and requested_regime is None:
        return _respond(
            message="Il regime selezionato non e' disponibile tra i corpora caricati.",
            sources=[],
            chat_id=payload.chat_id,
        )

    llm_client = _get_llm_client()
    if llm_client is None:
        return _respond(
            message=client_init_error or "Client DeepSeek non disponibile.",
            sources=[],
            regime_id=requested_regime.regime_id if requested_regime else None,
            chat_id=payload.chat_id,
        )

    if requested_regime is not None:
        active_regime, regime_explicit, regime_ambiguous = requested_regime, True, False
    else:
        active_regime, regime_explicit, regime_ambiguous = _resolve_regime(contenuto)
    if regime_ambiguous:
        available = ", ".join(profile.label for profile in REGIME_PROFILES)
        return _respond(
            message=(
                "La domanda sembra riferirsi a piu' regimi. Specifica meglio il regime fiscale da usare "
                f"tra quelli caricati: {available}."
            ),
            sources=[],
            chat_id=payload.chat_id,
        )
    if active_regime is None:
        return _respond(
            message=_regime_scope_message(),
            sources=[],
            chat_id=payload.chat_id,
        )

    # LLM-based query classification
    relevance = _classify_query_relevance(raw_contenuto, active_regime.label)
    if relevance == "off_topic":
        return ChatResponse(
            message=(
                f"{_regime_scope_message(active_regime)} "
                "Riformula la domanda in questo ambito."
            ),
            sources=[],
        )

    # ATECO deterministic lookup — triggered by LLM classification
    if relevance == "ateco_lookup" and active_regime.regime_id == "forfettario":
        ateco_data = _extract_ateco_components(contenuto)
        if ateco_data is not None:
            prefix, subcode = ateco_data
            coeff = _lookup_coefficiente_ateco(prefix, subcode=subcode)
            if coeff is not None:
                if subcode is not None and coeff.endswith("%"):
                    message = (
                        f"Il coefficiente di redditività per ATECO {prefix}.{subcode} è {coeff}."
                    )
                elif prefix == 46 or prefix == 47:
                    message = (
                        f"Per il codice ATECO {prefix}, {coeff} "
                        "Controlla sempre il sottocodice completo per il valore esatto."
                    )
                else:
                    message = (
                        f"Il coefficiente di redditività per ATECO {prefix} è {coeff}."
                    )
                return ChatResponse(
                    message=message,
                    sources=[
                        "03_Tabella_Coefficienti_Redditivita_ATECO.pdf",
                        "01_Legge_190-2014_Base_Normativa_e_Coefficienti.pdf",
                    ],
                )
        # If extraction failed, fall through to RAG+LLM

    definition_term = _extract_definition_term(raw_contenuto)
    term_mentions: List[LexicalChunk] = []
    if definition_term:
        term_mentions = _collect_term_mentions(definition_term, active_regime.regime_id)

    retrieved, retrieval_mode = _search_with_intent(
        contenuto, regime_id=active_regime.regime_id
    )

    # If RAG found nothing but lexical search found the term in documents,
    # use those mentions as context for the LLM instead of blocking with a
    # static "non viene definito" message.
    if not retrieved and definition_term and term_mentions:
        _log_rag_event(
            "definition_fallback_to_llm",
            {
                "query": raw_contenuto,
                "regime": active_regime.regime_id,
                "term": definition_term,
                "mention_count": len(term_mentions),
            },
        )
        retrieved = [
            RetrievedChunk(
                regime=chunk.regime,
                source=chunk.source,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=0.15,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
            )
            for chunk in term_mentions[:8]
        ]
        retrieval_mode = "lexical_fallback"

    if not retrieved:
        _log_rag_event(
            "rag_no_results",
            {"query": raw_contenuto, "regime": active_regime.regime_id},
        )
        return _respond(
            message=(
                f"Non trovo informazioni pertinenti nei documenti caricati per {active_regime.label.lower()}. "
                "Riformula la domanda o aggiungi documentazione."
            ),
            sources=[],
            regime_id=active_regime.regime_id,
            chat_id=payload.chat_id,
        )


    top_score = max(item.score for item in retrieved)
    if top_score < 0.12:
        _log_rag_event(
            "rag_low_confidence",
            {
                "query": raw_contenuto,
                "regime": active_regime.regime_id,
                "top_score": top_score,
                "sources": list(dict.fromkeys(item.source for item in retrieved))[:4],
            },
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
        f"Sei un assistente fiscale per {active_regime.label.lower()} in Italia. "
        "Rispondi solo con informazioni presenti nel CONTEXT. "
        "Se il CONTEXT non contiene una parte della risposta, dillo in una sola frase breve. "
        "Se un termine è solo citato ma non definito, dillo esplicitamente. "
        "Non affermare che un termine non è menzionato se compare nel CONTEXT. "
        "Non inventare norme, soglie o scadenze. "
        "Stile obbligatorio: italiano chiaro, tono professionale, nessun markdown, "
        "nessun uso di **, # o elenchi con trattini. "
        "Non iniziare con formule tipo 'In base al CONTEXT fornito'."
    )

    try:
        response = llm_client.chat.completions.create(
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
                        "Non inserire mai le fonti nel testo della risposta."
                    ),
                },
            ],
            stream=False,
        )
    except RateLimitError:
        return _respond(
            message=(
                "Quota DeepSeek esaurita o limite raggiunto (errore 429). "
                "Controlla piano e billing del provider selezionato."
            ),
            sources=[],
            regime_id=active_regime.regime_id,
            chat_id=payload.chat_id,
        )
    except APIError as error:
        return _respond(
            message=f"Errore API DeepSeek: {error}",
            sources=[],
            regime_id=active_regime.regime_id,
            chat_id=payload.chat_id,
        )

    answer = _clean_model_answer(response.choices[0].message.content or "")
    sources = list(dict.fromkeys(item.source for item in retrieved))[:4]
    source_details = _build_source_details(retrieved)
    confidence_label, confidence_score = _confidence_from_results(
        retrieved,
        retrieval_mode,
    )
    return _respond(
        message=answer,
        sources=sources,
        source_details=source_details,
        confidence_label=confidence_label,
        confidence_score=confidence_score,
        retrieval_mode=retrieval_mode,
        regime_id=active_regime.regime_id,
        chat_id=payload.chat_id,
    )

@app.post("/chat-stream")
async def chat_stream(payload: ChatRequest):
    """Streaming chat endpoint that returns Server-Sent Events."""
    import asyncio

    if SEMANTIC_SEARCH_ENABLED and not _ensure_rag_ready():
        async def error_gen():
            yield f"data: {json.dumps({'error': 'RAG non disponibile'}, ensure_ascii=False)}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    raw_contenuto = (payload.content or "").strip()
    if not raw_contenuto:
        async def empty_gen():
            yield f"data: {json.dumps({'error': 'Messaggio vuoto'}, ensure_ascii=False)}\n\n"
        return StreamingResponse(empty_gen(), media_type="text/event-stream")

    # Resolve regime (same logic as main endpoint)
    requested_regime = _resolve_requested_regime(payload.regime_id)
    if payload.regime_id and requested_regime is None:
        async def regime_notfound_gen():
            msg = "Il regime selezionato non e' disponibile tra i corpora caricati."
            yield f"data: {json.dumps({'text': msg, 'done': True, 'sources': []}, ensure_ascii=False)}\n\n"
        return StreamingResponse(regime_notfound_gen(), media_type="text/event-stream")

    contenuto = _normalize_tax_query(raw_contenuto)

    if requested_regime is not None:
        active_regime, regime_explicit, regime_ambiguous = requested_regime, True, False
    else:
        active_regime, regime_explicit, regime_ambiguous = _resolve_regime(contenuto)

    if regime_ambiguous:
        available = ", ".join(profile.label for profile in REGIME_PROFILES)
        async def ambiguous_gen():
            msg = f"La domanda sembra riferirsi a piu' regimi. Specifica meglio il regime fiscale tra: {available}."
            yield f"data: {json.dumps({'text': msg, 'done': True, 'sources': []}, ensure_ascii=False)}\n\n"
        return StreamingResponse(ambiguous_gen(), media_type="text/event-stream")

    if active_regime is None:
        async def noregime_gen():
            yield f"data: {json.dumps({'text': _regime_scope_message(), 'done': True, 'sources': []}, ensure_ascii=False)}\n\n"
        return StreamingResponse(noregime_gen(), media_type="text/event-stream")

    # LLM-based relevance classification
    relevance = _classify_query_relevance(raw_contenuto, active_regime.label)
    if relevance == "off_topic":
        async def offtopic_gen():
            msg = f"{_regime_scope_message(active_regime)} Riformula la domanda in questo ambito."
            yield f"data: {json.dumps({'text': msg, 'done': True, 'sources': []}, ensure_ascii=False)}\n\n"
        return StreamingResponse(offtopic_gen(), media_type="text/event-stream")

    # ATECO deterministic lookup — triggered by LLM classification
    if relevance == "ateco_lookup" and active_regime.regime_id == "forfettario":
        ateco_data = _extract_ateco_components(contenuto)
        if ateco_data is not None:
            prefix, subcode = ateco_data
            coeff = _lookup_coefficiente_ateco(prefix, subcode=subcode)
            if coeff is not None:
                if subcode is not None and coeff.endswith("%"):
                    msg = f"Il coefficiente di redditivita' per ATECO {prefix}.{subcode} e' {coeff}."
                elif prefix == 46 or prefix == 47:
                    msg = f"Per il codice ATECO {prefix}, {coeff}. Controlla sempre il sottocodice completo per il valore esatto."
                else:
                    msg = f"Il coefficiente di redditivita' per ATECO {prefix} e' {coeff}."
                async def ateco_gen():
                    yield f"data: {json.dumps({'text': msg, 'done': True, 'sources': ['03_Tabella_Coefficienti_Redditivita_ATECO.pdf', '01_Legge_190-2014_Base_Normativa_e_Coefficienti.pdf']}, ensure_ascii=False)}\n\n"
                return StreamingResponse(ateco_gen(), media_type="text/event-stream")
        # If extraction failed, fall through to RAG+LLM

    # RAG retrieval (same as main endpoint)
    retrieved, retrieval_mode = _search_with_intent(raw_contenuto, regime_id=active_regime.regime_id)

    if not retrieved:
        async def noresult_gen():
            msg = (
                "Non ho trovato informazioni pertinenti nei documenti disponibili. "
                "Prova a riformulare con termini piú specifici o a verificare la tua domanda."
            )
            yield f"data: {json.dumps({'text': msg, 'done': True, 'sources': []}, ensure_ascii=False)}\n\n"
        return StreamingResponse(noresult_gen(), media_type="text/event-stream")

    # Check confidence
    top_score = max(item.score for item in retrieved)
    if top_score < 0.12:
        _log_rag_event(
            "rag_low_confidence",
            {"query": raw_contenuto, "regime": active_regime.regime_id, "top_score": top_score},
        )

    # Build context
    context_blocks = []
    for item in retrieved:
        context_blocks.append(
            f"[Fonte: {item.source} | Chunk: {item.chunk_id} | Score: {item.score:.3f}]\n{item.text}"
        )
    context = "\n\n".join(context_blocks)

    system_prompt = (
        f"Sei un assistente fiscale per {active_regime.label.lower()} in Italia. "
        "Rispondi solo con informazioni presenti nel CONTEXT. "
        "Se il CONTEXT non contiene una parte della risposta, dillo in una sola frase breve. "
        "Non inventare norme, soglie o scadenze. "
        "Stile obbligatorio: italiano chiaro, tono professionale, nessun markdown, "
        "nessun uso di **, # o elenchi con trattini. "
        "Non iniziare con formule tipo 'In base al CONTEXT fornito'."
    )

    user_prompt = (
        f"DOMANDA:\n{contenuto}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "Rispondi in italiano in modo sintetico. "
        "Usa massimo 4 frasi totali. "
        "Dai prima la risposta diretta. "
        "Non inserire mai le fonti nel testo della risposta."
    )

    llm_client = _get_llm_client()
    if not llm_client:
        async def nollm_gen():
            yield f"data: {json.dumps({'error': client_init_error or 'LLM non disponibile'}, ensure_ascii=False)}\n\n"
        return StreamingResponse(nollm_gen(), media_type="text/event-stream")

    sources = list(dict.fromkeys(item.source for item in retrieved))[:4]
    source_details = _build_source_details(retrieved)
    confidence_label, confidence_score = _confidence_from_results(retrieved, retrieval_mode)

    async def stream_generator():
        try:
            response_stream = llm_client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
            )

            full_text = ""
            for chunk in response_stream:
                delta = chunk.choices[0].delta.content or "" if chunk.choices else ""
                if delta:
                    full_text += delta
                    yield f"data: {json.dumps({'chunk': delta}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0)  # Yield control

            # Final message with metadata - convert SourceRef to dict for JSON serialization
            final_payload = {
                "done": True,
                "text": _clean_model_answer(full_text),
                "sources": sources,
                "source_details": [
                    {
                        "source": d.source,
                        "excerpt": d.excerpt,
                        "chunk_id": d.chunk_id,
                        "page_start": d.page_start,
                        "page_end": d.page_end,
                        "score": d.score,
                    }
                    for d in source_details
                ],
                "confidence_label": confidence_label,
                "confidence_score": confidence_score,
                "retrieval_mode": retrieval_mode,
                "regime_id": active_regime.regime_id,
                "chat_id": payload.chat_id,
            }
            yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"

        except RateLimitError:
            yield f"data: {json.dumps({'error': 'Quota DeepSeek esaurita (errore 429)'}, ensure_ascii=False)}\n\n"
        except APIError as error:
            yield f"data: {json.dumps({'error': f'Errore API DeepSeek: {error}'}, ensure_ascii=False)}\n\n"
        except Exception as error:
            yield f"data: {json.dumps({'error': f'Errore interno: {error}'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

@app.get("/{asset_name}", include_in_schema=False, response_class=FileResponse)
async def serve_frontend_asset(asset_name: str):
    if asset_name not in FRONTEND_PAGES | FRONTEND_ASSETS:
        raise HTTPException(status_code=404, detail="Risorsa non trovata.")
    target_path = FRONTEND_ROOT / asset_name
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Risorsa non trovata.")
    return FileResponse(target_path)
