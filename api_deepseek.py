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
from fastapi.responses import FileResponse
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
HARD_CODED_MODE = os.getenv("HARD_CODED_MODE", "all").strip().lower()
HARD_CODED_CATEGORIES = {
    "all": {"critical", "stable", "optional"},
    "balanced": {"critical", "stable"},
    "critical": {"critical"},
}
ALLOWED_HARD_CODED = HARD_CODED_CATEGORIES.get(HARD_CODED_MODE, {"critical", "stable"})
chat_store = ChatHistoryStore(DATA_ROOT / "chat_history")
feedback_store = FeedbackStore(DATA_ROOT / "feedback" / "feedback.jsonl")
event_store = EventStore(DATA_ROOT / "events" / "app_events.jsonl")
ADMIN_ACCESS_KEY = os.getenv("ADMIN_ACCESS_KEY", "").strip()
FRONTEND_PAGES = {"index.html", "chat.html", "admin.html", "admin_tools.html"}
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


def _allow_hardcoded(category: str) -> bool:
    return category in ALLOWED_HARD_CODED


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

def _contains_percent_reference(query: str, value: str) -> bool:
    return re.search(rf"(?<!\d){re.escape(value)}\s*%", query) is not None


def _has_inps_35_context(query: str) -> bool:
    q = _normalize_tax_query(query)
    return any(
        term in q
        for term in (
            "riduzione contributiva",
            "riduzione inps",
            "riduzione del 35",
            "riduzione 35",
            "agevolazione",
            "artigiani",
            "commercianti",
            "gestione separata",
            "cassa professionale",
            "professionisti con cassa",
            "inps",
        )
    ) or _contains_percent_reference(q, "35")


def _has_non_inps_domain_context(query: str) -> bool:
    q = _normalize_tax_query(query)
    return any(
        term in q
        for term in (
            "ateco",
            "vies",
            "intrastat",
            "bollo",
            "extra-ue",
            "extra ue",
            "reverse",
            "td17",
            "google ads",
            "facebook ads",
            "meta ads",
            "lavoro dipendente",
            "reddito dipendente",
            "srl",
            "societa",
            "società",
            "ex datore",
            "partecipazioni",
            "residenza estera",
            "regimi speciali",
            "acconto",
            "saldo",
        )
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


def _is_ateco_coeff_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_ateco_ref = "ateco" in q or re.search(r"\bcodice\s*[0-9]{2}", q) is not None
    return (
        has_ateco_ref
        and (
            re.search(r"coeffic", q) is not None
            or re.search(r"reddi+tivit", q) is not None
        )
    )


def _is_ateco_list_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    return (
        "ateco" in q
        and (
            "tutti" in q
            or "elenco" in q
            or "quali sono" in q
            or "tabella" in q
        )
        and (
            "coeffic" in q
            or re.search(r"reddi+tivit", q) is not None
            or "codici" in q
        )
    )


def _is_ateco_codes_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    return "ateco" in q and (
        "tutti i codici" in q
        or "elenco codici" in q
        or "quali sono tutti i codici" in q
        or "codici ateco" in q
        or "elenco ateco" in q
        or "quali sono i codici" in q
    )


def _is_random_ateco_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_ateco = "ateco" in q
    has_random_intent = any(
        term in q
        for term in (
            "a caso",
            "random",
            "uno a caso",
            "qualsiasi",
            "dammi un codice",
        )
    )
    return has_ateco and has_random_intent


def _is_quadro_lm_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    return "quadro lm" in q or ("lm" in q and "quadro" in q)


def _is_off_topic_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    return any(
        term in q
        for term in (
            "sanremo",
            "meteo",
            "bitcoin",
            "barzelletta",
            "api key",
            ".env",
            "fuori tema",
            "chi ha vinto",
        )
    )


def _is_limit_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    return any(term in q for term in ("quanto posso guadagn", "limite", "soglia", "ricavi", "compensi"))


def _is_tax_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    return any(term in q for term in ("quanto vengo tass", "tassat", "imposta", "aliquota", "sostitutiva", "tasse"))


def _is_forfettario_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    return "forfett" in q or _query_mentions_regime_id(query, "forfettario")


def _is_tax_regime_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    return any(
        term in q
        for term in (
            "regime",
            "forfett",
            "fisco",
            "fiscal",
            "imposta",
            "aliquota",
            "iva",
            "contribut",
            "reddito",
            "imponib",
            "ricavi",
            "compensi",
            "soglia",
            "fattur",
            "detra",
            "dedu",
            "residenza",
            "rientr",
            "uscit",
            "apertura",
            "attivita",
            "partecipaz",
            "srl",
            "societ",
            "scadenz",
            "acconto",
            "saldo",
            "partita iva",
            "requisit",
            "accesso",
            "uscita",
            "ademp",
            "dichiar",
        )
    )


def _is_forfettario_intro_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_forfettario = _is_forfettario_query(q)
    has_intro_intent = any(
        term in q
        for term in (
            "cos'e",
            "cos e",
            "che cos'e",
            "che cos e",
            "spiegami",
            "spiegare",
            "in parole semplici",
            "in pratica",
            "come funziona",
            "come funziona in generale",
            "di cosa si tratta",
        )
    )
    technical_terms = (
        "ateco",
        "vies",
        "intrastat",
        "td17",
        "bollo",
        "inps",
        "35%",
        "soglia",
        "ricavi",
        "compensi",
        "srl",
        "ex datore",
        "lavoro dipendente",
        "cassa professionale",
        "google ads",
        "uscita",
        "scadenza",
        "acconto",
        "saldo",
        "causa ostativa",
        "cause ostative",
        "regimi speciali",
    )
    return has_forfettario and has_intro_intent and not any(term in q for term in technical_terms)


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


def _is_vies_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    return (
        "vies" in q
        or (
            any(term in q for term in ("intracomunit", "unione europea", "ue"))
            and any(term in q for term in ("iscrizion", "obbligator", "quando serve", "a cosa serve"))
        )
    )


def _is_bollo_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_bollo = "bollo" in q
    has_estero = any(term in q for term in ("estera", "estere", "estero", "extra-ue", "extra ue", "ue"))
    has_threshold = any(term in q for term in ("77,47", "77.47", "2 euro", "2,00"))
    return has_bollo and (has_estero or has_threshold)


def _is_eu_b2b_services_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    if "b2c" in q:
        return False
    has_service = "serviz" in q or "b2b" in q
    has_eu_context = any(
        term in q
        for term in (
            "b2b",
            "cliente ue",
            "unione europea",
            "intracomunit",
            "verso ue",
            "nell'ue",
            "in ue",
        )
    )
    has_invoice_intent = any(term in q for term in ("fattur", "dicitura", "iva", "come", "vendita"))
    return has_service and has_eu_context and has_invoice_intent


def _is_extra_ue_services_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_service = "serviz" in q
    has_extra_context = any(
        term in q
        for term in (
            "extra-ue",
            "extra ue",
            "usa",
            "fuori ue",
            "cliente estero",
            "verso estero",
        )
    )
    has_invoice_intent = any(term in q for term in ("fattur", "dicitura", "iva", "come", "vendita"))
    return has_service and has_extra_context and has_invoice_intent


def _is_eu_b2c_services_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_service = "serviz" in q
    has_b2c_context = any(
        term in q
        for term in (
            "b2c",
            "cliente privato ue",
            "privato ue",
            "consumatore ue",
        )
    )
    has_invoice_intent = any(term in q for term in ("fattur", "dicitura", "iva", "come", "vendita"))
    return has_service and has_b2c_context and has_invoice_intent


def _is_forfettario_exit_100k_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_threshold = any(term in q for term in ("100.000", "100000", "100k"))
    has_exit_intent = any(term in q for term in ("esco", "uscita", "subito", "anno dopo", "quando"))
    return has_threshold and has_exit_intent


def _is_cash_basis_threshold_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_cash_terms = any(
        term in q for term in ("fatturato", "incassato", "incassi", "criterio di cassa")
    )
    has_threshold_terms = any(
        term in q for term in ("soglie", "soglia", "limite", "ricavi", "compensi")
    )
    return has_cash_terms and has_threshold_terms


def _is_aliquota_5_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_aliquota = any(term in q for term in ("aliquota", "imposta sostitutiva")) or _contains_percent_reference(q, "5")
    has_five = _contains_percent_reference(q, "5") or "5 per cento" in q
    has_when = any(term in q for term in ("quando", "applica", "si applica"))
    return has_aliquota and has_five and has_when


def _is_ads_reverse_charge_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_ads = any(term in q for term in ("google ads", "facebook ads", "meta ads"))
    has_reverse_context = any(term in q for term in ("td17", "reverse", "iva", "autofattura"))
    return has_ads and has_reverse_context


def _is_intrastat_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    return "intrastat" in q


def _is_extra_ue_wording_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_extra_context = any(term in q for term in ("extra-ue", "extra ue", "fuori ue"))
    has_wording_intent = any(term in q for term in ("dicitura", "artt. 7", "7-septies", "articoli 7"))
    return has_extra_context and has_wording_intent


def _is_bollo_exact_threshold_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_threshold = "77,47" in q or "77.47" in q
    has_exact_intent = any(term in q for term in ("esatt", "uguale", "pari a", "preciso"))
    has_bollo = "bollo" in q
    return has_bollo and has_threshold and has_exact_intent


def _is_employment_income_threshold_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_employment_context = any(
        term in q
        for term in (
            "lavoro dipendente",
            "reddito da lavoro dipendente",
            "redditi da lavoro dipendente",
            "reddito dipendente",
            "come dipendente",
            "sono dipendente",
            "faccio il dipendente",
        )
    )
    has_threshold_or_access_intent = any(
        term in q
        for term in (
            "30.000",
            "30000",
            "trentamila",
            "posso stare",
            "posso rimanere",
            "accesso",
            "forfettario",
        )
    )
    return has_employment_context and has_threshold_or_access_intent


def _is_employment_income_under_threshold_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_employment_context = any(
        term in q
        for term in (
            "lavoro dipendente",
            "reddito da lavoro dipendente",
            "redditi da lavoro dipendente",
            "reddito dipendente",
            "dipendente",
        )
    )
    has_under_threshold = any(
        term in q
        for term in (
            "29.000",
            "29000",
            "29mila",
            "sotto 30.000",
            "inferiore a 30.000",
        )
    )
    has_access_intent = any(term in q for term in ("posso", "restare", "forfettario"))
    return has_employment_context and has_access_intent and has_under_threshold


def _is_employment_cessation_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_cessation = "cessat" in q
    has_employment_context = any(
        term in q for term in ("rapporto di lavoro", "lavoro dipendente", "dipendente")
    )
    has_effect_intent = any(term in q for term in ("cambia", "rileva", "forfettario", "soglia", "conta"))
    return has_cessation and has_employment_context and has_effect_intent


def _is_inps_35_general_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    strong_terms = (
        "riduzione contributiva",
        "riduzione inps",
        "riduzione del 35",
        "riduzione 35",
        "sconto del 35",
        "sconto inps",
    )
    follow_up_terms = (
        "domanda",
        "presentata",
        "entro quando",
        "decorre",
        "rinnova",
        "rinnovo",
        "automatic",
        "pensione",
        "rinuncio",
        "rinuncia",
        "agevolazione",
        "si perde",
        "nuove attività",
        "richiederla",
        "di nuovo",
        "riattiv",
        "chi può",
        "vale anche",
    )
    context_anchor_terms = (
        "inps",
        "riduzione contributiva",
        "riduzione",
        "35%",
        "contribut",
        "agevolazione",
        "artigiani",
        "commercianti",
        "rinnova",
        "rinuncio",
        "richiederla",
    )
    if any(term in q for term in strong_terms):
        return True
    if _contains_percent_reference(q, "35"):
        return True
    if any(term in q for term in ("ateco", "vies", "intrastat", "bollo", "extra-ue", "reverse", "td17")):
        return False
    return any(term in q for term in follow_up_terms) and any(
        term in q for term in context_anchor_terms
    )


def _is_forfettario_domain_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    domain_terms = (
        "forfett",
        "regime",
        "ateco",
        "iva",
        "inps",
        "contribut",
        "fattur",
        "bollo",
        "vies",
        "reverse charge",
        "inversione contabile",
        "intrastat",
        "b2b",
        "b2c",
        "extra-ue",
        "extra ue",
        "intracomunit",
        "cliente ue",
        "cliente estero",
        "vendita servizi",
        "dicitura",
        "ricavi",
        "compensi",
        "imposta",
        "aliquota",
        "sostitutiva",
        "soglia",
        "limite",
        "quadro lm",
        "artigiani",
        "commercianti",
        "partita iva",
        "scadenz",
        "acconto",
        "saldo",
        "cause ostative",
        "lavoro dipendente",
        "srl",
        "societa",
        "società",
        "partecipazioni",
        "controllo",
        "ex datore",
        "riduzione",
        "35%",
        "domanda",
        "decorre",
        "rinnova",
        "rinuncio",
        "rinuncia",
        "richiederla",
        "riattiv",
        "pensione",
        "gestione separata",
        "cassa professionale",
        "google ads",
        "facebook ads",
        "td17",
        "100k",
        "100.000",
        "100000",
        "uscita",
        "esco",
        "presentata",
        "detraz",
        "dedu",
        "figli a carico",
        "asilo nido",
        "bene strumentale",
        "beni strumentali",
        "plusvalenza",
        "rimborso",
        "730",
        "modello 730",
        "naspi",
        "residenza",
        "regimi speciali",
        "agricoltura",
        "editoria",
        "cassa professionale",
        "cassa forense",
        "inarcassa",
        "contributo integrativo",
    )
    return any(term in q for term in domain_terms)


def _is_inps_35_deadline_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_inps_context = _has_inps_35_context(q)
    has_deadline_intent = any(
        term in q
        for term in (
            "scadenz",
            "entro quando",
            "termine",
            "quando va presentata",
            "presentazione",
        )
    )
    return has_inps_context and has_deadline_intent


def _is_inps_35_short_deadline_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_inps_context = _has_inps_35_context(q)
    has_deadline_intent = any(
        term in q
        for term in (
            "entro quando",
            "scadenza",
            "quando va presentata",
            "termine",
        )
    )
    generic_follow_up = not _has_non_inps_domain_context(q)
    return has_deadline_intent and "domanda" in q and (has_inps_context or generic_follow_up)


def _is_inps_35_march_decorrenza_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_march_example = "10 marzo" in q or ("marzo" in q and any(term in q for term in ("domanda", "invio", "invi")))
    has_decorrenza_intent = any(
        term in q
        for term in (
            "decorre",
            "da quando",
            "quando decorre",
            "decorrenza",
        )
    )
    generic_follow_up = not _has_non_inps_domain_context(q)
    return has_march_example and has_decorrenza_intent and (
        _has_inps_35_context(q) or generic_follow_up
    )


def _is_inps_35_reapply_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_renounce = any(
        term in q
        for term in (
            "rinuncio",
            "rinuncia",
            "rinunciare",
            "rinunci",
            "revoca",
        )
    )
    has_reapply = any(
        term in q
        for term in (
            "richiederla",
            "richiedere di nuovo",
            "nuova domanda",
            "di nuovo",
            "riattiv",
        )
    )
    return has_renounce and has_reapply


def _is_inps_35_renewal_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_inps_35_context = _has_inps_35_context(q)
    has_renewal_intent = "rinnova" in q or "rinnovo" in q
    has_auto_intent = "automatic" in q or "ogni anno" in q
    return has_inps_35_context and has_renewal_intent and has_auto_intent


def _is_inps_35_cassa_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_35 = _contains_percent_reference(q, "35") or any(
        term in q for term in ("riduzione inps", "riduzione contributiva", "sconto inps", "sconto del 35")
    )
    has_cassa = "cassa" in q or "professionist" in q
    return has_35 and has_cassa


def _is_srl_control_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_srl = "srl" in q or "societa" in q or "società" in q
    has_control = any(term in q for term in ("controllo", "2359", "controllo di fatto"))
    has_forfettario_intent = any(
        term in q
        for term in (
            "forfett",
            "posso restare",
            "posso accedere",
            "causa ostativa",
            "restare",
            "accedere",
        )
    )
    return has_srl and has_control and has_forfettario_intent


def _is_inps_35_new_activity_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_new_activity = any(
        term in q
        for term in (
            "nuove attivita",
            "nuove attività",
            "nuova attivita",
            "nuova attività",
            "appena aperto",
            "inizio attivita",
            "inizio attività",
        )
    )
    has_request_intent = any(
        term in q
        for term in ("domanda", "richiesta", "quando va fatta", "quando chiedo", "quando la chiedo", "quando richiedo")
    )
    generic_follow_up = not _has_non_inps_domain_context(q)
    return has_new_activity and has_request_intent and (
        _has_inps_35_context(q) or generic_follow_up
    )


def _is_inps_35_apply_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_apply_intent = any(
        term in q
        for term in (
            "come si presenta",
            "come present",
            "come fare domanda",
            "dove si presenta",
            "dove fare domanda",
        )
    )
    generic_follow_up = "domanda" in q and not _has_non_inps_domain_context(q)
    return has_apply_intent and (_has_inps_35_context(q) or generic_follow_up)


def _is_inps_35_late_deadline_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_late_deadline = any(
        term in q
        for term in (
            "dopo il 28 febbraio",
            "dopo 28 febbraio",
            "oltre il 28 febbraio",
            "oltre 28 febbraio",
        )
    )
    return has_late_deadline and "domanda" in q and (
        _has_inps_35_context(q) or not _has_non_inps_domain_context(q)
    )


def _is_inps_35_loss_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_loss_intent = any(
        term in q for term in ("si perde", "quando si perde", "perdo", "perdita")
    )
    return _has_inps_35_context(q) and has_loss_intent


def _is_ex_datore_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_ex_datore = "ex datore" in q or "datore di lavoro" in q
    has_ostativa = any(term in q for term in ("causa ostativa", "ostativo", "forfettario"))
    return has_ex_datore and has_ostativa


def _is_ex_datore_after_two_years_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_ex_datore = _is_ex_datore_query(q)
    has_time_reference = any(
        term in q
        for term in (
            "3 anni",
            "tre anni",
            "oltre due anni",
            "piu di due anni",
            "più di due anni",
        )
    )
    return has_ex_datore and has_time_reference


def _is_business_meal_cost_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_meal_context = any(
        term in q
        for term in (
            "cena",
            "pranzo",
            "ristorante",
            "cliente a cena",
            "portato un cliente",
        )
    )
    has_tax_intent = any(
        term in q
        for term in (
            "scaricare l'iva",
            "scaricare iva",
            "scaricare l iva",
            "dedurre",
            "deduc",
            "abbassare le tasse",
            "15%",
        )
    )
    return has_meal_context and has_tax_intent


def _is_employee_above_threshold_access_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_employee_context = any(
        term in q
        for term in (
            "dipendente",
            "lavoro come dipendente",
            "lavoro dipendente",
            "reddito da lavoro dipendente",
        )
    )
    has_threshold = any(term in q for term in ("32.000", "32000", "30.000", "30000"))
    has_access_intent = any(
        term in q
        for term in (
            "aprire la partita iva",
            "aprire p.iva",
            "aprire partita iva",
            "posso aprire",
            "arrotondare",
            "forfettari",
            "forfettario",
        )
    )
    return has_employee_context and has_threshold and has_access_intent


def _is_ex_employer_prevalence_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_previous_employer = any(
        term in q
        for term in (
            "ex datore",
            "datore di lavoro",
            "ex azienda",
            "mia ex azienda",
            "mi sono licenziato",
            "licenziato",
        )
    )
    has_invoicing_intent = any(
        term in q
        for term in (
            "fatturare tutto",
            "fatturare",
            "prevalent",
            "tutto il mio lavoro",
            "risparmio sulle tasse",
        )
    )
    return has_previous_employer and has_invoicing_intent


def _is_strumental_asset_sale_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_asset_context = any(
        term in q
        for term in (
            "pc aziendale",
            "vecchio pc",
            "computer aziendale",
            "bene strumentale",
            "beni strumentali",
        )
    ) or ("venduto" in q and "pc" in q)
    has_threshold_intent = any(
        term in q
        for term in (
            "85.000",
            "85000",
            "85mila",
            "limite annuale",
            "limite dei ricavi",
            "sommare",
            "concorre",
        )
    )
    return has_asset_context and has_threshold_intent


def _is_family_detraction_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_family_context = any(
        term in q
        for term in (
            "figli a carico",
            "carico",
            "asilo nido",
            "spese d'istruzione",
            "spese di istruzione",
        )
    )
    has_detraction_intent = any(
        term in q
        for term in (
            "19%",
            "19 per cento",
            "detraz",
            "recuperare",
            "tasse della mia partita iva",
        )
    )
    return has_family_context and has_detraction_intent


def _is_exit_100k_example_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_superamento = any(term in q for term in ("105.000", "105000", "oltre 100.000", "oltre 100000"))
    has_timing_intent = any(
        term in q
        for term in (
            "resto forfettario fino a dicembre",
            "cambio l'anno prossimo",
            "anno prossimo",
            "fino a dicembre",
        )
    )
    return has_superamento and has_timing_intent


def _is_foreign_software_reverse_charge_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_software_context = any(
        term in q
        for term in (
            "software",
            "abbonamento software",
            "sito americano",
            "sito estero",
            "americano",
            "estero",
        )
    )
    has_vat_doubt = any(
        term in q
        for term in (
            "non c'e l'iva",
            "non c'è l'iva",
            "senza iva",
            "sono a posto",
            "a posto cosi",
            "a posto così",
        )
    )
    return has_software_context and has_vat_doubt


def _is_srl_non_reconducible_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_srl_context = "srl" in q and any(
        term in q for term in ("20%", "20 per cento", "20 %", "socio")
    )
    has_non_reconducible_example = any(
        term in q
        for term in (
            "pulizie",
            "marketing",
            "consulente marketing",
            "attivita diverse",
            "attivita non riconducibili",
        )
    )
    has_forfettario_intent = any(
        term in q
        for term in (
            "posso",
            "aprire",
            "forfettari",
            "forfettaria",
            "forfettario",
        )
    )
    return has_srl_context and has_non_reconducible_example and has_forfettario_intent


def _is_bollo_reimbursement_tax_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_bollo_context = "bollo" in q and any(
        term in q
        for term in (
            "rimborsato",
            "rimborso",
            "2 euro",
            "2€",
            "marca da bollo",
        )
    )
    has_tax_intent = any(
        term in q
        for term in (
            "pagare le tasse",
            "ci devo pagare le tasse",
            "tassabile",
            "tassato",
            "ricavo",
            "85.000",
            "85000",
        )
    )
    return has_bollo_context and has_tax_intent


def _is_residency_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_residency = any(
        term in q
        for term in (
            "residenza fiscale",
            "residente all'estero",
            "residente estero",
            "vivo all'estero",
            "vivo all estero",
            "all estero",
            "non residente",
            "residenza",
        )
    )
    has_forfettario_intent = any(
        term in q
        for term in ("forfettario", "forfettaria", "forfettari", "posso", "accedere", "applicare")
    )
    return has_residency and has_forfettario_intent


def _is_special_vat_regime_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_special_context = any(
        term in q
        for term in (
            "regimi speciali iva",
            "regime speciale iva",
            "agricoltura",
            "editoria",
            "agenzia di viaggio",
            "tabacchi",
            "sali e tabacchi",
        )
    )
    has_access_intent = any(
        term in q
        for term in ("forfettario", "compatibile", "posso", "accedere", "incompatibile")
    )
    return has_special_context and has_access_intent


def _is_730_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    return "730" in query or "modello 730" in q


def _is_730_only_forfettario_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_730 = _is_730_query(query)
    has_forfettario = _is_forfettario_query(q) or "partita iva" in q
    has_declare_intent = any(
        term in q
        for term in (
            "posso presentare",
            "posso fare",
            "posso usare",
            "dichiarare",
            "mettere",
            "quadro e",
            "detrazioni",
            "scaricare",
        )
    )
    return has_730 and has_forfettario and has_declare_intent


def _is_cassa_integrativo_threshold_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_cassa = any(
        term in q
        for term in (
            "cassa forense",
            "inarcassa",
            "enpam",
            "cipag",
            "cassa professionale",
            "contributo integrativo",
        )
    ) or _contains_percent_reference(q, "4") or _contains_percent_reference(q, "2") or _contains_percent_reference(q, "5")
    has_threshold_intent = any(
        term in q
        for term in (
            "85.000",
            "85000",
            "limite",
            "conta",
            "concorre",
            "in fattura",
        )
    )
    return has_cassa and has_threshold_intent


def _is_cassa_integrativo_deduction_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_integrativo = (
        "contributo integrativo" in q
        or _contains_percent_reference(q, "4")
        or _contains_percent_reference(q, "2")
        or _contains_percent_reference(q, "5")
    ) and any(term in q for term in ("cassa", "inarcassa", "forense", "enpam", "cipag", "professionale"))
    has_deduction_intent = any(
        term in q
        for term in (
            "deducibile",
            "dedurre",
            "rigo lm35",
            "ci pago le tasse",
            "tass",
        )
    )
    return has_integrativo and has_deduction_intent


def _is_naspi_anticipation_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_naspi = "naspi" in q
    has_anticipation = any(
        term in q
        for term in (
            "anticipo",
            "anticipata",
            "unica soluzione",
            "30 giorni",
            "apertura della partita iva",
        )
    )
    return has_naspi and has_anticipation


def _is_naspi_monthly_compatibility_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_naspi = "naspi" in q
    has_monthly_context = any(
        term in q
        for term in (
            "mensile",
            "compatibile",
            "continuare a ricevere",
            "ridotta",
            "naspi-com",
            "5500",
            "8500",
        )
    )
    return has_naspi and has_monthly_context


def _is_general_forfettario_tax_query(query: str) -> bool:
    q = _normalize_tax_query(query)
    has_forfettario = _is_forfettario_query(q)
    has_tax_intent = any(
        term in q
        for term in ("che tasse pago", "quali tasse", "imposta sostitutiva", "aliquota", "quanto pago")
    )
    has_limit_terms = any(term in q for term in ("soglia", "limite", "85.000", "100.000", "ricavi", "compensi"))
    return has_forfettario and has_tax_intent and not has_limit_terms


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

    is_forfettario_regime = active_regime.regime_id == "forfettario"
    allow_critical = _allow_hardcoded("critical")
    allow_stable = _allow_hardcoded("stable")
    allow_optional = _allow_hardcoded("optional")

    if _is_off_topic_query(contenuto):
        return ChatResponse(
            message=(
                f"{_regime_scope_message(active_regime)} "
                "Riformula la domanda in questo ambito."
            ),
            sources=[],
        )

    if allow_optional and is_forfettario_regime and _is_forfettario_intro_query(contenuto):
        return ChatResponse(
            message=(
                "Il regime forfettario è un regime fiscale agevolato per partite IVA individuali. "
                "In pratica, il reddito imponibile non si calcola sottraendo tutte le spese reali una per una, "
                "ma applicando ai ricavi un coefficiente di redditività legato all'attività svolta. "
                "Su quel reddito si paga di norma un'imposta sostitutiva del 15%, che in alcuni casi scende al 5% "
                "per le nuove attività. Prevede anche semplificazioni IVA e contabili, ma si può usare solo se "
                "rispetti i requisiti e non hai cause ostative."
            ),
            sources=[
                "01_Legge_190-2014_Base_Normativa_e_Coefficienti.pdf",
                "04_Elenco_Cause_Ostative_e_Esclusioni_2026.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_quadro_lm_query(contenuto):
        return ChatResponse(
            message=(
                "Il Quadro LM è la sezione del Modello Redditi Persone Fisiche dedicata ai contribuenti "
                "che applicano il regime forfettario. Serve a determinare il reddito imponibile e "
                "l'imposta sostitutiva dovuta."
            ),
            sources=[
                "09_Guida_Tecnica_Quadro_LM_Dichiarazione_Redditi.pdf",
            ],
        )

    if allow_critical and is_forfettario_regime and _is_ateco_coeff_query(contenuto):
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
        if _is_ateco_list_query(contenuto):
            return ChatResponse(
                message=(
                    "Nel regime forfettario i coefficienti di redditività sono associati a gruppi ATECO: "
                    "40% (es. commercio e alloggio/ristorazione), "
                    "54% (commercio ambulante alimenti/bevande), "
                    "62% (intermediari del commercio), "
                    "67% (altre attività economiche), "
                    "78% (attività professionali, sanitarie, istruzione e finanziarie), "
                    "86% (costruzioni e attività immobiliari). "
                    "Se mi indichi un codice ATECO specifico, ti dico il coefficiente esatto."
                ),
                sources=[
                    "03_Tabella_Coefficienti_Redditivita_ATECO.pdf",
                    "01_Legge_190-2014_Base_Normativa_e_Coefficienti.pdf",
                ],
            )

    if allow_critical and is_forfettario_regime and _is_random_ateco_query(contenuto):
        return ChatResponse(
            message=(
                "Non posso inventare un codice ATECO a caso. Nei documenti disponibili non c'è "
                "l'elenco completo dei codici ATECO italiani; c'è solo la tabella dei gruppi ATECO "
                "con i relativi coefficienti. Se mi dai un codice specifico, posso dirti il coefficiente."
            ),
            sources=[
                "03_Tabella_Coefficienti_Redditivita_ATECO.pdf",
                "01_Legge_190-2014_Base_Normativa_e_Coefficienti.pdf",
            ],
        )

    if allow_critical and is_forfettario_regime and _is_ateco_codes_query(contenuto):
        return ChatResponse(
            message=(
                "Nei documenti disponibili non c'è un elenco completo di tutti i codici ATECO italiani; "
                "c'è la tabella dei gruppi ATECO rilevanti per il regime forfettario con i relativi coefficienti. "
                "Se vuoi, posso dirti il coefficiente partendo da un codice ATECO preciso."
            ),
            sources=[
                "03_Tabella_Coefficienti_Redditivita_ATECO.pdf",
                "01_Legge_190-2014_Base_Normativa_e_Coefficienti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_forfettario_query(contenuto) and _is_limit_query(contenuto) and _is_tax_query(contenuto):
        return ChatResponse(
            message=(
                "Per restare nel regime forfettario, nel periodo precedente ricavi o compensi non devono superare 85.000 euro; "
                "se durante l'anno superi 100.000 euro, l'uscita dal regime è immediata. "
                "L'imposta sostitutiva è in via ordinaria al 15%, ridotta al 5% per i primi 5 anni se sono rispettati i requisiti di nuova attività."
            ),
            sources=[
                "02_Circolare_32E-2023_Novita_Soglie_e_Uscita_Immediat.pdf",
                "01_Legge_190-2014_Base_Normativa_e_Coefficienti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_forfettario_query(contenuto) and _is_limit_query(contenuto):
        return ChatResponse(
            message=(
                "Per il regime forfettario, la soglia ordinaria è 85.000 euro di ricavi o compensi. "
                "L'uscita immediata scatta se nell'anno superi 100.000 euro."
            ),
            sources=[
                "02_Circolare_32E-2023_Novita_Soglie_e_Uscita_Immediat.pdf",
                "01_Legge_190-2014_Base_Normativa_e_Coefficienti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_general_forfettario_tax_query(contenuto):
        return ChatResponse(
            message=(
                "Nel regime forfettario paghi un'imposta sostitutiva che in via ordinaria e' del 15%. "
                "Per le nuove attivita', se rispetti i requisiti, l'aliquota scende al 5% per i primi 5 anni."
            ),
            sources=[
                "01_Legge_190-2014_Base_Normativa_e_Coefficienti.pdf",
                "09_Guida_Tecnica_Quadro_LM_Dichiarazione_Redditi.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_forfettario_exit_100k_query(contenuto):
        return ChatResponse(
            message=(
                "Se superi 100.000 euro di ricavi o compensi nell'anno, l'uscita dal forfettario è immediata "
                "dal momento del superamento."
            ),
            sources=[
                "02_Circolare_32E-2023_Novita_Soglie_e_Uscita_Immediat.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_cash_basis_threshold_query(contenuto):
        return ChatResponse(
            message=(
                "Per le soglie del regime forfettario conta quanto incassi, non quanto fatturi, perché si applica "
                "il criterio di cassa. "
                "Quindi per verificare i limiti di 85.000 e 100.000 euro rilevano i ricavi o compensi percepiti."
            ),
            sources=[
                "02_Circolare_32E-2023_Novita_Soglie_e_Uscita_Immediat.pdf",
                "01_Legge_190-2014_Base_Normativa_e_Coefficienti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_aliquota_5_query(contenuto):
        return ChatResponse(
            message=(
                "L'aliquota del 5% si applica alle nuove attività che rispettano i requisiti previsti "
                "per il regime forfettario, e vale per i primi 5 anni."
            ),
            sources=[
                "01_Legge_190-2014_Base_Normativa_e_Coefficienti.pdf",
                "02_Circolare_32E-2023_Novita_Soglie_e_Uscita_Immediat.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_inps_35_apply_query(contenuto):
        return ChatResponse(
            message=(
                "La domanda per la riduzione INPS del 35% è esclusivamente telematica. "
                "Si presenta dal portale INPS, con accesso SPID, CIE o CNS, nel Cassetto Previdenziale per "
                "Artigiani e Commercianti alla voce Domande telematizzate e Regime agevolato."
            ),
            sources=[
                "11_Guida_Riduzione_Contributiva_35_INPS.pdf",
                "10_Circolare_INPS_14-2026_Artigiani_e_Commercianti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_inps_35_deadline_query(contenuto):
        return ChatResponse(
            message=(
                "La domanda per la riduzione contributiva INPS del 35% si presenta solo online nel Cassetto "
                "Previdenziale Artigiani e Commercianti (accesso con SPID/CIE/CNS). "
                "Per i contribuenti già attivi va presentata entro il 28 febbraio di ogni anno; "
                "se inviata dopo, l'agevolazione decorre dal 1° gennaio dell'anno successivo. "
                "Per le nuove attività, la richiesta va fatta tempestivamente dopo l'iscrizione previdenziale."
            ),
            sources=[
                "11_Guida_Riduzione_Contributiva_35_INPS.pdf",
                "10_Circolare_INPS_14-2026_Artigiani_e_Commercianti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_inps_35_late_deadline_query(contenuto):
        return ChatResponse(
            message=(
                "Sì, puoi presentarla anche dopo il 28 febbraio, ma per i contribuenti già attivi la riduzione "
                "non opera nell'anno in corso. "
                "Se la domanda è tardiva, l'agevolazione decorre dal 1° gennaio dell'anno successivo."
            ),
            sources=[
                "11_Guida_Riduzione_Contributiva_35_INPS.pdf",
                "10_Circolare_INPS_14-2026_Artigiani_e_Commercianti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_inps_35_short_deadline_query(contenuto):
        return ChatResponse(
            message=(
                "Per i contribuenti già attivi, la domanda per la riduzione INPS del 35% va presentata "
                "entro il 28 febbraio di ogni anno."
            ),
            sources=[
                "11_Guida_Riduzione_Contributiva_35_INPS.pdf",
                "10_Circolare_INPS_14-2026_Artigiani_e_Commercianti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_inps_35_march_decorrenza_query(contenuto):
        return ChatResponse(
            message=(
                "Se presenti la domanda il 10 marzo, la riduzione non decorre nell'anno in corso ma dal "
                "1° gennaio dell'anno successivo."
            ),
            sources=[
                "11_Guida_Riduzione_Contributiva_35_INPS.pdf",
                "10_Circolare_INPS_14-2026_Artigiani_e_Commercianti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_inps_35_new_activity_query(contenuto):
        return ChatResponse(
            message=(
                "Per una nuova attività, la richiesta della riduzione INPS del 35% va fatta dopo l'iscrizione "
                "alla Gestione Artigiani e Commercianti, tramite il Cassetto Previdenziale. "
                "Va presentata tempestivamente, senza attendere il 28 febbraio dell'anno successivo."
            ),
            sources=[
                "11_Guida_Riduzione_Contributiva_35_INPS.pdf",
                "10_Circolare_INPS_14-2026_Artigiani_e_Commercianti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_inps_35_reapply_query(contenuto):
        return ChatResponse(
            message=(
                "Sì, se rinunci puoi presentare una nuova domanda in seguito, purché tu sia ancora nel "
                "regime forfettario e restino i requisiti per l'agevolazione. "
                "Per i contribuenti già attivi si applica il termine ordinario del 28 febbraio; se presenti "
                "la domanda dopo tale data, la riduzione decorre dal 1° gennaio dell'anno successivo."
            ),
            sources=[
                "11_Guida_Riduzione_Contributiva_35_INPS.pdf",
                "10_Circolare_INPS_14-2026_Artigiani_e_Commercianti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_inps_35_renewal_query(contenuto):
        return ChatResponse(
            message=(
                "Sì, l'agevolazione si rinnova se continui ad avere i requisiti del regime forfettario e "
                "dell'iscrizione alla Gestione Artigiani e Commercianti. "
                "Se i requisiti vengono meno o rinunci, si torna al regime contributivo ordinario."
            ),
            sources=[
                "11_Guida_Riduzione_Contributiva_35_INPS.pdf",
                "10_Circolare_INPS_14-2026_Artigiani_e_Commercianti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_inps_35_loss_query(contenuto):
        return ChatResponse(
            message=(
                "L'agevolazione del 35% si perde se non restano i requisiti per applicarla o se presenti "
                "rinuncia. In quel caso si rientra nel regime contributivo ordinario."
            ),
            sources=[
                "11_Guida_Riduzione_Contributiva_35_INPS.pdf",
                "10_Circolare_INPS_14-2026_Artigiani_e_Commercianti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_inps_35_cassa_query(contenuto):
        return ChatResponse(
            message=(
                "No, la riduzione INPS del 35% non si applica ai professionisti iscritti a una Cassa "
                "professionale e non si applica neppure alla Gestione Separata. "
                "È riservata agli imprenditori individuali forfettari iscritti alla Gestione Artigiani e Commercianti."
            ),
            sources=[
                "11_Guida_Riduzione_Contributiva_35_INPS.pdf",
                "10_Circolare_INPS_14-2026_Artigiani_e_Commercianti.pdf",
                "06_Circolare_INPS_8-2026_Aliquote_Gestione_Separata.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_inps_35_general_query(contenuto):
        return ChatResponse(
            message=(
                "La riduzione INPS del 35% è riservata agli imprenditori individuali forfettari iscritti alla "
                "Gestione Artigiani e Commercianti; non si applica a Gestione Separata o Casse professionali. "
                "La domanda è telematica; per gli attivi va presentata entro il 28 febbraio, se tardiva decorre "
                "dal 1° gennaio dell'anno successivo. "
                "L'agevolazione si rinnova se restano i requisiti, ma riduce la contribuzione utile ai fini pensionistici."
            ),
            sources=[
                "11_Guida_Riduzione_Contributiva_35_INPS.pdf",
                "10_Circolare_INPS_14-2026_Artigiani_e_Commercianti.pdf",
                "06_Circolare_INPS_8-2026_Aliquote_Gestione_Separata.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_ex_datore_after_two_years_query(contenuto):
        return ChatResponse(
            message=(
                "No, non è ostativo se il rapporto con l'ex datore di lavoro è cessato da oltre due periodi "
                "d'imposta. La causa ostativa riguarda la fatturazione prevalente verso datori di lavoro con "
                "cui il rapporto è in corso o è cessato nei due precedenti periodi d'imposta."
            ),
            sources=[
                "04_Elenco_Cause_Ostative_e_Esclusioni_2026.pdf",
                "05_Circolare_9E-2019_Approfondimento_Cause_Ostative.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_srl_control_query(contenuto):
        return ChatResponse(
            message=(
                "In presenza di controllo, anche di fatto, una partecipazione in SRL può costituire causa "
                "ostativa se la società esercita attività economiche direttamente o indirettamente riconducibili "
                "a quella svolta individualmente. La sola percentuale di partecipazione non basta: conta il controllo "
                "ai sensi dell'articolo 2359 c.c. e la riconducibilità dell'attività."
            ),
            sources=[
                "04_Elenco_Cause_Ostative_e_Esclusioni_2026.pdf",
                "05_Circolare_9E-2019_Approfondimento_Cause_Ostative.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_vies_query(contenuto):
        return ChatResponse(
            message=(
                "Sì, per il forfettario l'iscrizione al VIES è necessaria quando effettua operazioni "
                "intracomunitarie (vendita di servizi o acquisto di beni/servizi nell'UE). "
                "Serve a operare con partita IVA abilitata nei rapporti UE."
            ),
            sources=[
                "12_Operazioni_Estere_VIES_Reverse_Charge_e_Dogane.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_ads_reverse_charge_query(contenuto):
        return ChatResponse(
            message=(
                "Per acquisti di servizi esteri come Google Ads o Facebook Ads, in forfettario si applica "
                "integrazione/autofattura TD17 con IVA al 22% e versamento entro il giorno 16 del mese successivo "
                "con F24 codice 6099. L'IVA resta un costo non detraibile."
            ),
            sources=[
                "12_Operazioni_Estere_VIES_Reverse_Charge_e_Dogane.pdf",
                "08a_Guida_Pratica_Fatturazione_Elettronica_Forfettari_2026.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_bollo_reimbursement_tax_query(contenuto):
        return ChatResponse(
            message=(
                "No. Il rimborso dei 2 euro del bollo da parte del cliente non costituisce un ricavo "
                "aggiuntivo da tassare separatamente e non si somma al limite degli 85.000 euro. Il bollo "
                "resta solo un riaddebito dell'imposta assolta in fattura."
            ),
            sources=[
                "08b_Manuale_AdE_Imposta_Bollo_Fatture_Elettroniche.pdf",
                "08a_Guida_Pratica_Fatturazione_Elettronica_Forfettari_2026.pdf",
                "13_CASI CRITICI E RISPOSTE AI QUESITI (PRASSI ADE).pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_residency_query(contenuto):
        return ChatResponse(
            message=(
                "In via generale il regime forfettario e' riservato ai residenti in Italia. Fanno eccezione "
                "i residenti in uno Stato UE o SEE con adeguato scambio di informazioni, ma solo se producono "
                "in Italia almeno il 75% del reddito complessivo."
            ),
            sources=[
                "04_Elenco_Cause_Ostative_e_Esclusioni_2026.pdf",
                "01_Legge_190-2014_Base_Normativa_e_Coefficienti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_special_vat_regime_query(contenuto):
        return ChatResponse(
            message=(
                "No, i regimi speciali IVA sono incompatibili con il regime forfettario. Se ti avvali di "
                "regimi speciali come agricoltura, editoria, agenzie di viaggio o sali e tabacchi, non puoi "
                "accedere o permanere nel forfettario."
            ),
            sources=[
                "04_Elenco_Cause_Ostative_e_Esclusioni_2026.pdf",
                "01_Legge_190-2014_Base_Normativa_e_Coefficienti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_730_only_forfettario_query(contenuto):
        return ChatResponse(
            message=(
                "Il reddito della partita IVA forfettaria non si dichiara nel 730 ma nel Modello Redditi PF, "
                "quadro LM. Il 730 puoi usarlo solo per eventuali altri redditi soggetti a IRPEF, come lavoro "
                "dipendente, pensione, terreni o fabbricati; se hai solo reddito forfettario, il 730 non basta "
                "e le detrazioni si usano solo se hai capienza IRPEF su altri redditi."
            ),
            sources=[
                "16_IL MODELLO 730 E IL CONTRIBUENTE FORFETTARIO (2026).pdf",
                "09_Guida_Tecnica_Quadro_LM_Dichiarazione_Redditi.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_cassa_integrativo_threshold_query(contenuto):
        return ChatResponse(
            message=(
                "No. Il contributo integrativo addebitato in fattura dagli iscritti a una Cassa professionale "
                "(per esempio 2%, 4% o 5%) non concorre al reddito forfettario e non conta nel limite degli "
                "85.000 euro."
            ),
            sources=[
                "15_CASSE PROFESSIONALI AUTONOME E REGIME FORFETTARIO (2026).pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_cassa_integrativo_deduction_query(contenuto):
        return ChatResponse(
            message=(
                "No. Il contributo integrativo non e' deducibile e non va trattato come costo del professionista, "
                "perche' e' un importo addebitato al cliente e poi riversato alla Cassa. In deduzione, nel rigo "
                "LM35, rilevano invece i contributi soggettivi e maternita' effettivamente versati."
            ),
            sources=[
                "15_CASSE PROFESSIONALI AUTONOME E REGIME FORFETTARIO (2026).pdf",
                "09_Guida_Tecnica_Quadro_LM_Dichiarazione_Redditi.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_naspi_anticipation_query(contenuto):
        return ChatResponse(
            message=(
                "Sì, chi apre una partita IVA forfettaria puo' chiedere l'anticipazione della NASPI in un'unica "
                "soluzione, ma la domanda va inviata all'INPS entro 30 giorni dall'apertura della partita IVA o "
                "dall'inizio attivita'. L'importo anticipato e' tassato ordinariamente IRPEF, non rientra nel "
                "forfettario e non concorre al limite degli 85.000 euro."
            ),
            sources=[
                "14_NASPI, DISOCCUPAZIONE E INCENTIVI ALL'AUTOIMPRENDITORIALITÀ (2026).pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_naspi_monthly_compatibility_query(contenuto):
        return ChatResponse(
            message=(
                "La NASPI mensile puo' essere compatibile con la partita IVA forfettaria, ma non resta intera: "
                "va comunicato il reddito presunto all'INPS con NASPI-COM entro un mese e l'assegno viene ridotto. "
                "Nei documenti la compatibilita' e' indicata entro 5.500 euro per attivita' d'impresa/commercio e "
                "8.500 euro per lavoro autonomo o professionale."
            ),
            sources=[
                "14_NASPI, DISOCCUPAZIONE E INCENTIVI ALL'AUTOIMPRENDITORIALITÀ (2026).pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_bollo_exact_threshold_query(contenuto):
        return ChatResponse(
            message=(
                "No, con importo esattamente pari a 77,47 euro il bollo non si applica. "
                "L'imposta di bollo da 2,00 euro scatta solo oltre 77,47 euro."
            ),
            sources=[
                "08b_Manuale_AdE_Imposta_Bollo_Fatture_Elettroniche.pdf",
                "08a_Guida_Pratica_Fatturazione_Elettronica_Forfettari_2026.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_bollo_query(contenuto):
        return ChatResponse(
            message=(
                "Sì, se la fattura supera 77,47 euro si applica l'imposta di bollo da 2,00 euro, "
                "anche nelle fatture verso l'estero. "
                "In fattura va indicata la dicitura di assolvimento del bollo; "
                "il versamento è gestito con liquidazione periodica tramite i canali dell'Agenzia delle Entrate."
            ),
            sources=[
                "08b_Manuale_AdE_Imposta_Bollo_Fatture_Elettroniche.pdf",
                "08a_Guida_Pratica_Fatturazione_Elettronica_Forfettari_2026.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_intrastat_query(contenuto):
        return ChatResponse(
            message=(
                "Nei documenti, l'Intrastat è richiamato per le operazioni di servizi B2B verso soggetti UE. "
                "Per i casi specifici (periodicità e obbligo puntuale) è necessaria verifica operativa sul singolo caso."
            ),
            sources=[
                "12_Operazioni_Estere_VIES_Reverse_Charge_e_Dogane.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_eu_b2c_services_query(contenuto):
        return ChatResponse(
            message=(
                "Nei documenti disponibili è trattata in modo esplicito soprattutto la casistica B2B UE "
                "(reverse charge e Intrastat). "
                "Per servizi B2C verso cliente UE la disciplina IVA dipende dal tipo di servizio e dal luogo di consumo, "
                "quindi serve una verifica specifica prima di emettere fattura."
            ),
            sources=[
                "12_Operazioni_Estere_VIES_Reverse_Charge_e_Dogane.pdf",
                "08a_Guida_Pratica_Fatturazione_Elettronica_Forfettari_2026.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_eu_b2b_services_query(contenuto):
        return ChatResponse(
            message=(
                "Per servizi B2B verso cliente UE, la fattura si emette senza IVA con dicitura "
                "\"Reverse Charge\" o \"Inversione contabile\" e richiede iscrizione VIES. "
                "Per queste operazioni è previsto l'adempimento Intrastat. "
                "Se la fattura supera 77,47 euro, si applica anche il bollo da 2,00 euro."
            ),
            sources=[
                "12_Operazioni_Estere_VIES_Reverse_Charge_e_Dogane.pdf",
                "08a_Guida_Pratica_Fatturazione_Elettronica_Forfettari_2026.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_extra_ue_wording_query(contenuto):
        return ChatResponse(
            message=(
                "La dicitura da usare per servizi extra-UE è: "
                "\"Operazione non soggetta ai sensi degli artt. da 7 a 7-septies del DPR 633/72\"."
            ),
            sources=[
                "12_Operazioni_Estere_VIES_Reverse_Charge_e_Dogane.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_extra_ue_services_query(contenuto):
        return ChatResponse(
            message=(
                "Per servizi verso cliente extra-UE, la fattura è senza IVA con dicitura: "
                "\"Operazione non soggetta ai sensi degli artt. da 7 a 7-septies del DPR 633/72\". "
                "Se l'importo supera 77,47 euro, si applica il bollo da 2,00 euro."
            ),
            sources=[
                "12_Operazioni_Estere_VIES_Reverse_Charge_e_Dogane.pdf",
                "08b_Manuale_AdE_Imposta_Bollo_Fatture_Elettroniche.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_employment_income_under_threshold_query(contenuto):
        return ChatResponse(
            message=(
                "Con reddito da lavoro dipendente pari a 29.000 euro, la sola soglia dei 30.000 euro "
                "non è causa ostativa. "
                "Restano comunque da verificare le altre condizioni di accesso e permanenza nel forfettario."
            ),
            sources=[
                "04_Elenco_Cause_Ostative_e_Esclusioni_2026.pdf",
                "05_Circolare_9E-2019_Approfondimento_Cause_Ostative.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_employment_income_threshold_query(contenuto):
        return ChatResponse(
            message=(
                "In generale, con redditi da lavoro dipendente o assimilati superiori a 30.000 euro "
                "non puoi applicare il regime forfettario. "
                "La soglia non rileva se il rapporto di lavoro è cessato. "
                "Restano comunque da verificare anche le altre cause ostative."
            ),
            sources=[
                "04_Elenco_Cause_Ostative_e_Esclusioni_2026.pdf",
                "05_Circolare_9E-2019_Approfondimento_Cause_Ostative.pdf",
                "02_Circolare_32E-2023_Novita_Soglie_e_Uscita_Immediat.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_employment_cessation_query(contenuto):
        return ChatResponse(
            message=(
                "Sì: se il rapporto di lavoro dipendente è cessato, la soglia dei 30.000 euro non rileva "
                "come causa ostativa. Restano però da verificare gli altri requisiti del regime forfettario."
            ),
            sources=[
                "04_Elenco_Cause_Ostative_e_Esclusioni_2026.pdf",
                "05_Circolare_9E-2019_Approfondimento_Cause_Ostative.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_business_meal_cost_query(contenuto):
        return ChatResponse(
            message=(
                "No. Nel regime forfettario non detrai l'IVA sugli acquisti e non deduci analiticamente "
                "spese come una cena con cliente. Il vantaggio fiscale e' gia' forfettizzato tramite il "
                "coefficiente di redditivita', quindi quella fattura non ti abbassa separatamente l'imposta del 15%."
            ),
            sources=[
                "01_Legge_190-2014_Base_Normativa_e_Coefficienti.pdf",
                "08a_Guida_Pratica_Fatturazione_Elettronica_Forfettari_2026.pdf",
                "13_CASI CRITICI E RISPOSTE AI QUESITI (PRASSI ADE).pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_employee_above_threshold_access_query(contenuto):
        return ChatResponse(
            message=(
                "No, in via generale non puoi applicare il regime forfettario se hai redditi da lavoro "
                "dipendente o assimilati superiori a 30.000 euro. La soglia diventa irrilevante solo se il "
                "rapporto di lavoro e' cessato; altrimenti questa e' una causa ostativa."
            ),
            sources=[
                "04_Elenco_Cause_Ostative_e_Esclusioni_2026.pdf",
                "05_Circolare_9E-2019_Approfondimento_Cause_Ostative.pdf",
                "16_IL MODELLO 730 E IL CONTRIBUENTE FORFETTARIO (2026).pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_ex_employer_prevalence_query(contenuto):
        return ChatResponse(
            message=(
                "Puoi aprire la partita IVA, ma c'e' un rischio forte di causa ostativa se fatturi "
                "prevalentemente all'ex datore di lavoro o alla ex azienda con cui il rapporto e' in corso "
                "o e' cessato nei due precedenti periodi d'imposta. Se oltre il 50% dei compensi arriva da "
                "quel soggetto, perdi il forfettario dall'anno successivo."
            ),
            sources=[
                "04_Elenco_Cause_Ostative_e_Esclusioni_2026.pdf",
                "05_Circolare_9E-2019_Approfondimento_Cause_Ostative.pdf",
                "13_CASI CRITICI E RISPOSTE AI QUESITI (PRASSI ADE).pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_strumental_asset_sale_query(contenuto):
        return ChatResponse(
            message=(
                "No. La cessione di un bene strumentale usato, come un vecchio PC aziendale, non si somma "
                "ai ricavi o compensi che contano per la soglia degli 85.000 euro. In questi casi il "
                "forfettario non tassa la plusvalenza come ricavo ordinario del regime."
            ),
            sources=[
                "13_CASI CRITICI E RISPOSTE AI QUESITI (PRASSI ADE).pdf",
                "01_Legge_190-2014_Base_Normativa_e_Coefficienti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_family_detraction_query(contenuto):
        return ChatResponse(
            message=(
                "No. L'imposta sostitutiva del regime forfettario non consente di recuperare dal carico "
                "fiscale della partita IVA le detrazioni per figli a carico o spese come l'asilo nido. "
                "Quelle agevolazioni non riducono l'imposta sostitutiva del forfettario; restano solo gli "
                "eventuali strumenti dedicati, come l'Assegno Unico, se spettanti."
            ),
            sources=[
                "16_IL MODELLO 730 E IL CONTRIBUENTE FORFETTARIO (2026).pdf",
                "01_Legge_190-2014_Base_Normativa_e_Coefficienti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_exit_100k_example_query(contenuto):
        return ChatResponse(
            message=(
                "No, non resti forfettario fino a dicembre. Se superi 100.000 euro di compensi o ricavi "
                "percepiti nell'anno, l'uscita dal regime e' immediata dal momento del superamento e "
                "sull'operazione che fa superare la soglia devi applicare subito il regime IVA ordinario."
            ),
            sources=[
                "02_Circolare_32E-2023_Novita_Soglie_e_Uscita_Immediat.pdf",
                "13_CASI CRITICI E RISPOSTE AI QUESITI (PRASSI ADE).pdf",
                "01_Legge_190-2014_Base_Normativa_e_Coefficienti.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_foreign_software_reverse_charge_query(contenuto):
        return ChatResponse(
            message=(
                "No, non sei a posto cosi'. Se acquisti un software o un servizio digitale da un fornitore "
                "estero senza IVA, devi emettere autofattura/integrazione TD17 con IVA italiana al 22% e "
                "versarla con F24 entro il giorno 16 del mese successivo. Nel forfettario quell'IVA resta "
                "un costo e non e' detraibile."
            ),
            sources=[
                "12_Operazioni_Estere_VIES_Reverse_Charge_e_Dogane.pdf",
                "08a_Guida_Pratica_Fatturazione_Elettronica_Forfettari_2026.pdf",
            ],
        )

    if allow_stable and is_forfettario_regime and _is_srl_non_reconducible_query(contenuto):
        return ChatResponse(
            message=(
                "Sì, in linea di principio puoi applicare il forfettario se hai solo il 20% della SRL, "
                "quindi senza controllo, e l'attivita' individuale non e' riconducibile a quella della "
                "societa'. Nel tuo esempio pulizie e consulenza marketing sono attivita' diverse, quindi "
                "la sola partecipazione del 20% non integra di per se' la causa ostativa."
            ),
            sources=[
                "04_Elenco_Cause_Ostative_e_Esclusioni_2026.pdf",
                "05_Circolare_9E-2019_Approfondimento_Cause_Ostative.pdf",
                "13_CASI CRITICI E RISPOSTE AI QUESITI (PRASSI ADE).pdf",
            ],
        )

    if not regime_explicit and is_forfettario_regime and not _is_forfettario_domain_query(contenuto):
        return ChatResponse(
            message=(
                f"{_regime_scope_message(active_regime)} "
                "Riformula la domanda in questo ambito."
            ),
            sources=[],
        )

    if regime_explicit and not _is_tax_regime_query(contenuto):
        return ChatResponse(
            message=(
                f"{_regime_scope_message(active_regime)} "
                "Riformula la domanda in questo ambito."
            ),
            sources=[],
        )

    definition_term = _extract_definition_term(raw_contenuto)
    term_mentions: List[LexicalChunk] = []
    if definition_term:
        term_mentions = _collect_term_mentions(definition_term, active_regime.regime_id)

    retrieved, retrieval_mode = _search_with_intent(
        contenuto, regime_id=active_regime.regime_id
    )
    if not retrieved:
        if definition_term and term_mentions:
            _log_rag_event(
                "definition_cited_not_defined",
                {
                    "query": raw_contenuto,
                    "regime": active_regime.regime_id,
                    "term": definition_term,
                    "sources": list(dict.fromkeys(chunk.source for chunk in term_mentions))[:4],
                },
            )
            return _respond(
                message=_definition_fallback_message(definition_term),
                sources=list(dict.fromkeys(chunk.source for chunk in term_mentions))[:4],
                regime_id=active_regime.regime_id,
                chat_id=payload.chat_id,
            )
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

    if definition_term:
        has_term_in_context = any(
            _term_appears_in_text(definition_term, item.text) for item in retrieved
        )
        if retrieval_mode == "lexical" and (term_mentions or has_term_in_context):
            sources = (
                list(dict.fromkeys(chunk.source for chunk in term_mentions))[:4]
                if term_mentions
                else list(dict.fromkeys(item.source for item in retrieved))[:4]
            )
            _log_rag_event(
                "definition_cited_not_defined",
                {
                    "query": raw_contenuto,
                    "regime": active_regime.regime_id,
                    "term": definition_term,
                    "sources": sources,
                },
            )
            return _respond(
                message=_definition_fallback_message(definition_term),
                sources=sources,
                regime_id=active_regime.regime_id,
                chat_id=payload.chat_id,
            )
        if not has_term_in_context and term_mentions:
            _log_rag_event(
                "definition_cited_not_defined",
                {
                    "query": raw_contenuto,
                    "regime": active_regime.regime_id,
                    "term": definition_term,
                    "sources": list(dict.fromkeys(chunk.source for chunk in term_mentions))[:4],
                },
            )
            return _respond(
                message=_definition_fallback_message(definition_term),
                sources=list(dict.fromkeys(chunk.source for chunk in term_mentions))[:4],
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


@app.get("/{asset_name}", include_in_schema=False, response_class=FileResponse)
async def serve_frontend_asset(asset_name: str):
    if asset_name not in FRONTEND_PAGES | FRONTEND_ASSETS:
        raise HTTPException(status_code=404, detail="Risorsa non trovata.")
    target_path = FRONTEND_ROOT / asset_name
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Risorsa non trovata.")
    return FileResponse(target_path)
